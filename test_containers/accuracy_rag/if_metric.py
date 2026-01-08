"""
Instruction-Following Metric

This module provides programmatic evaluation of whether LLM responses
follow given instruction constraints without using LLMs.
"""

import json
from typing import Any, Dict, Optional

from openai import OpenAI

# Import instruction classes and registry
from instructions_registry import INSTRUCTION_DICT


class InstructionFollowingMetric:
    """Evaluates whether responses follow instruction constraints"""

    def __init__(
        self, llm_model: str = "gpt-4o-mini", openai_client: Optional[OpenAI] = None
    ):
        """Initialize the metric

        Args:
            llm_model: The LLM model to use for semantic validation
            openai_client: OpenAI client instance for semantic validation (optional)
        """
        self.llm_model = llm_model
        self.openai_client = openai_client

    def _build_detailed_reason(self, success, instruction_type, response, metadata):
        """Fallback reason builder for classes that don't implement their own reason generation.

        Args:
            success: Boolean indicating if the instruction was followed
            instruction_type: Type of instruction (json, words, sentences, two_responses)
            response: The actual response from the model
            metadata: Instruction metadata dictionary

        Returns:
            Generic reason string
        """
        return "Instruction followed" if success else "Instruction not followed"

    def evaluate(
        self, response: str, instruction_type: str, instruction_type_metadata: str
    ) -> Dict[str, Any]:
        """
        Evaluate if a response follows the instruction constraint.

        Args:
            response: The LLM response to evaluate
            instruction_type: The type of instruction (e.g., "json", "words", "sentences", "two_responses")
            instruction_type_metadata: JSON string with instruction metadata

        Returns:
            Dict with keys:
                - score: Float between 0.0 and 1.0
                - success: Boolean indicating if constraint was followed
                - reason: String explaining the evaluation
        """
        try:
            metadata = json.loads(instruction_type_metadata)
        except json.JSONDecodeError as e:
            return {
                "score": 0.0,
                "success": False,
                "reason": f"Invalid metadata JSON: {str(e)}",
            }

        instruction_id = metadata.get("instruction_id")

        # Get the instruction class from INSTRUCTION_DICT
        if instruction_id not in INSTRUCTION_DICT:
            return {
                "score": 0.0,
                "success": False,
                "reason": f"Unknown instruction type: {instruction_id}",
            }

        instruction_class = INSTRUCTION_DICT[instruction_id]

        try:
            # Instantiate the instruction object
            # Most instructions may use the OpenAI client for format detection via LLM
            if instruction_id == "combination:two_responses":
                instruction_obj = instruction_class(
                    instruction_id,
                    openai_client=self.openai_client,
                    llm_model=self.llm_model,
                )
            elif instruction_id in [
                "length_constraints:number_sentences",
                "length_constraints:number_words",
            ]:
                # Pass OpenAI client for LLM-based format detection
                instruction_obj = instruction_class(
                    instruction_id,
                    openai_client=self.openai_client,
                    llm_model=self.llm_model,
                )
            else:
                instruction_obj = instruction_class(instruction_id)

            # Extract metadata args to pass to check_following
            check_args = {
                k: v for k, v in metadata.items() if k not in ["type", "instruction_id"]
            }

            # Call check_following to validate response with the parameters
            # The method now returns (success, reason) tuple
            result = instruction_obj.check_following(response, **check_args)

            # Handle both old bool return and new tuple return for backward compatibility
            if isinstance(result, tuple) and len(result) == 2:
                success, reason = result
            elif isinstance(result, bool):
                # Fallback for classes that still return just bool
                success = result
                reason = self._build_detailed_reason(
                    success, instruction_type, response, metadata
                )
            else:
                return {
                    "score": 0.0,
                    "success": False,
                    "reason": f"Invalid return type from check_following: {type(result)}",
                }

            # Check if this is an invalid format sample (marked in reason)
            is_invalid_format = "INVALID_FORMAT" in (reason or "")

            # Provide a clearer, client-facing explanation about invalid format status
            if is_invalid_format:
                invalid_format_reason = (
                    "The response does not conform to the required "
                    "instruction format (plain text unless specifically instructed otherwise). Such records "
                    "should be excluded from aggregate metric computations and reviewed manually."
                )
            else:
                invalid_format_reason = "Response format appears valid for evaluating the score for the requested instruction."

            # Return evaluation result (keep 'reason' as the short pass/fail message)
            score = 1.0 if success else 0.0

            return {
                "score": score,
                "success": success,
                "reason": reason,
                "is_invalid_format": is_invalid_format,
                "invalid_format_reason": invalid_format_reason,
            }

        except Exception as e:
            import traceback

            return {
                "score": 0.0,
                "success": False,
                "reason": f"Evaluation error: {str(e)}",
                "is_invalid_format": False,
                "invalid_format_reason": "Evaluation failed due to an internal error; see traceback for details.",
                "traceback": traceback.format_exc(),
            }


def evaluate_instruction_following(
    response: str, instruction_type: str, instruction_type_metadata: str
) -> Dict[str, Any]:
    """
    Standalone function to evaluate instruction following.

    Args:
        response: The LLM response to evaluate
        instruction_type: The type of instruction
        instruction_type_metadata: JSON string with instruction metadata

    Returns:
        Evaluation result dictionary
    """
    metric = InstructionFollowingMetric()
    return metric.evaluate(response, instruction_type, instruction_type_metadata)
