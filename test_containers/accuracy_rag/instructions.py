# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of instructions."""

import json
from typing import Optional

from openai import OpenAI
import instructions_util
import re

# The relational operations for comparison - Sentences supports three types
_COMPARISON_RELATION_SENTENCES = ("less than", "at least", "exactly")

# The relational operations for comparison - Words supports two types
_COMPARISON_RELATION_WORDS = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 3

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


def _parse_llm_json_response(response_content: str):
    """Parse LLM response that may be wrapped in markdown code blocks.

    Args:
      response_content: Raw response from LLM (may contain markdown code blocks)

    Returns:
      Parsed JSON object (dict or list)

    Raises:
      json.JSONDecodeError: If content cannot be parsed as JSON
      ValueError: If content is empty or malformed
    """
    if not response_content or not isinstance(response_content, str):
        raise ValueError(
            f"Expected string response content, got {type(response_content)}"
        )

    content = response_content.strip()
    if not content:
        raise ValueError("Response content is empty after stripping whitespace")

    # Strip markdown code blocks if present (multiple potential formats)
    original_content = content

    # Try to remove markdown fencing (multiple attempts to be safe)
    for fence in ["```json", "```JSON", "```Json"]:
        if content.startswith(fence):
            content = content[len(fence) :]
            break

    # Also remove generic code block fence
    if content.startswith("```"):
        content = content[3:]

    # Remove closing fence
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    if not content:
        raise ValueError(
            f"Content is empty after removing markdown fences. Original: {original_content[:100]}"
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Provide detailed error message for debugging
        raise json.JSONDecodeError(
            f"Failed to parse JSON. Error: {e.msg} at line {e.lineno}, col {e.colno}. "
            f"Content (first 200 chars): {content[:200]}",
            content,
            e.pos,
        )


def _is_structured_format(
    value: str, openai_client: Optional[OpenAI] = None, llm_model: str = "gpt-4o-mini"
) -> bool:
    """Check if response is in structured format (JSON, code, etc.) vs plain text using LLM.

    Args:
      value: The response text to check
      openai_client: OpenAI client for LLM detection (if None, uses quick heuristic)
      llm_model: The LLM model to use for format detection

    Returns:
      True if response is in structured/code format, False if plain text
    """
    # If no client provided, use quick heuristic as fallback
    if openai_client is None:
        value_stripped = value.strip()
        if value_stripped.startswith("```json") or value_stripped.startswith("```JSON"):
            return True
        if (value_stripped.startswith("{") and value_stripped.endswith("}")) or (
            value_stripped.startswith("[") and value_stripped.endswith("]")
        ):
            try:
                json.loads(value_stripped)
                return True
            except (ValueError, json.JSONDecodeError):
                pass
        if "```" in value_stripped:
            return True
        return False

    try:
        # Use LLM to classify format
        prompt = f"""Is the following response in plain English text format, or is it in a structured format like JSON, code, XML, or other technical format?

Response:
{value[:2000]}

Answer with ONLY one word: either "plain" (for regular English text) or "structured" (for JSON, code blocks, XML, etc.)

Answer:"""

        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=10,
            max_tokens=5,
        )

        answer = response.choices[0].message.content.strip().lower()
        is_structured = "structured" in answer

        return is_structured

    except Exception as e:
        # If LLM call fails, fall back to heuristic
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"LLM format detection failed ({type(e).__name__}), falling back to heuristic"
        )

        # Quick heuristic fallback
        value_stripped = value.strip()
        if value_stripped.startswith("```json") or value_stripped.startswith("```JSON"):
            return True
        if (value_stripped.startswith("{") and value_stripped.endswith("}")) or (
            value_stripped.startswith("[") and value_stripped.endswith("]")
        ):
            try:
                json.loads(value_stripped)
                return True
            except (ValueError, json.JSONDecodeError):
                pass
        if "```" in value_stripped:
            return True
        return False


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    def __init__(
        self,
        instruction_id,
        openai_client: Optional[OpenAI] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize with optional OpenAI client for format detection.

        Args:
          instruction_id: The instruction ID
          openai_client: OpenAI client instance for format detection via LLM
          llm_model: The LLM model to use for format detection
        """
        super().__init__(instruction_id)
        self.openai_client = openai_client
        self.llm_model = llm_model

    def check_following(self, value, num_sentences=None, relation=None):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.
          num_sentences: An integer specifying the number of sentences as a threshold.
          relation: A string in (`less than`, `at least`, `exactly`), defining the relational operator.

        Returns:
          Tuple of (success: bool, reason: str)

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`, `exactly`].
        """
        # Set defaults if not provided
        if num_sentences is None or num_sentences < 0:
            num_sentences = 1  # Default fallback
        if relation is None:
            relation = "at least"  # Default fallback
        elif relation not in _COMPARISON_RELATION_SENTENCES:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION_SENTENCES}, but {relation} is given."
            )

        # Check if format is structured (JSON, code, etc.) - mark as invalid
        is_structured = _is_structured_format(value, self.openai_client, self.llm_model)
        if is_structured:
            return (
                False,
                "INVALID_FORMAT: Response is in structured format (JSON/code). Sentence counting requires plain text. Sample marked as invalid and excluded from scoring.",
            )

        num_sentences_actual = instructions_util.count_sentences(value)
        success = False

        if relation == _COMPARISON_RELATION_SENTENCES[0]:  # "less than"
            success = num_sentences_actual < num_sentences
        elif relation == _COMPARISON_RELATION_SENTENCES[1]:  # "at least"
            success = num_sentences_actual >= num_sentences
        elif relation == _COMPARISON_RELATION_SENTENCES[2]:  # "exactly"
            success = num_sentences_actual == num_sentences

        # Generate reason
        if success:
            reason = f"Instruction followed: Sentence count constraint satisfied (expected: {relation} {num_sentences} sentences, actual: {num_sentences_actual})"
        else:
            reason = f"Instruction not followed: Sentence count constraint violated (expected: {relation} {num_sentences} sentences, actual: {num_sentences_actual})"

        return success, reason


class NumberOfWords(Instruction):
    """Checks the number of words."""

    def __init__(
        self,
        instruction_id,
        openai_client: Optional[OpenAI] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize with optional OpenAI client for format detection.

        Args:
          instruction_id: The instruction ID
          openai_client: OpenAI client instance for format detection via LLM
          llm_model: The LLM model to use for format detection
        """
        super().__init__(instruction_id)
        self.openai_client = openai_client
        self.llm_model = llm_model

    def check_following(self, value, num_words=None, relation=None):
        """Checks if the response contains the expected number of words.

        Args:
          value: A string representing the response.
          num_words: An integer specifying the number of words contained in the response.
          relation: A string in (`less than`, `at least`), defining the relational operator.

        Returns:
          Tuple of (success: bool, reason: str)
        """
        # Set defaults if not provided
        if num_words is None or num_words < 0:
            num_words = 100  # Default fallback
        if relation is None:
            relation = "at least"  # Default fallback
        elif relation not in _COMPARISON_RELATION_WORDS:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION_WORDS}, but {relation} is given."
            )

        # Check if format is structured (JSON, code, etc.) - mark as invalid
        is_structured = _is_structured_format(value, self.openai_client, self.llm_model)
        if is_structured:
            return (
                False,
                "INVALID_FORMAT: Response is in structured format (JSON/code). Word counting requires plain text. Sample marked as invalid and excluded from scoring.",
            )

        num_words_actual = instructions_util.count_words(value)
        success = False

        if relation == _COMPARISON_RELATION_WORDS[0]:
            success = num_words_actual < num_words
        elif relation == _COMPARISON_RELATION_WORDS[1]:
            success = num_words_actual >= num_words

        # Generate reason
        if success:
            reason = f"Instruction followed: Word count constraint satisfied (expected: {relation} {num_words} words, actual: {num_words_actual})"
        else:
            reason = f"Instruction not followed: Word count constraint violated (expected: {relation} {num_words} words, actual: {num_words_actual})"

        return success, reason


class JsonFormat(Instruction):
    """Check the Json format."""

    def __init__(self, instruction_id):
        super().__init__(instruction_id)

    def check_following(self, value):
        """Check if response is valid JSON format.

        Returns:
          Tuple of (success: bool, reason: str)
        """
        original_value = value
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
            return True, "Instruction followed: Response is valid JSON format"
        except ValueError as _:
            return False, self._generate_json_failure_reason(original_value)

    def _generate_json_failure_reason(self, response):
        """Generate detailed reason for JSON format failure."""
        response_stripped = response.strip()

        # Check if response has markdown code blocks
        has_markdown_blocks = "```" in response
        has_json_marker = (
            "```json" in response.lower()
            or "```Json" in response
            or "```JSON" in response
        )

        # Check if response starts with { or [
        starts_with_json = response_stripped.startswith(
            "{"
        ) or response_stripped.startswith("[")

        # Diagnose the reason
        if has_markdown_blocks:
            if has_json_marker:
                # Has ```json markers but invalid JSON
                return "Instruction not followed: Response has ```json markdown but contains invalid JSON"
            else:
                # Has markdown blocks but not JSON-specific
                return "Instruction not followed: Response has markdown code blocks ``` but no JSON content"
        elif starts_with_json:
            # Looks like JSON but isn't valid
            return "Instruction not followed: Response starts with JSON syntax but is not valid JSON"
        else:
            # Plain text, no JSON markers
            return "Instruction not followed: Response is plain text, not JSON format"


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def __init__(
        self,
        instruction_id,
        openai_client: Optional[OpenAI] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize with optional OpenAI client for semantic checking.

        Args:
          instruction_id: The instruction ID
          openai_client: OpenAI client instance for semantic validation (if None, uses fallback heuristic)
          llm_model: The LLM model to use for semantic difference validation
        """
        super().__init__(instruction_id)
        self.openai_client = openai_client
        self.llm_model = llm_model

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          Tuple of (success: bool, reason: str)
        """
        # Check if the delimiter exists in the response
        if "******" not in value:
            return (
                False,
                "Instruction not followed: No '******' delimiter found - responses not properly separated",
            )

        # Check delimiter count
        delimiter_count = value.count("******")
        if delimiter_count != 1:
            return (
                False,
                f"Instruction not followed: Found {delimiter_count} delimiters, expected exactly 1",
            )

        # Split by the delimiter
        parts = value.split("******")

        # Collect non-empty responses
        valid_responses = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                valid_responses.append(stripped)

        # Must have exactly 2 non-empty responses
        if len(valid_responses) != 2:
            return (
                False,
                f"Instruction not followed: Found {len(valid_responses)} non-empty response(s), expected 2",
            )

        # We only check for delimiter and two non-empty responses here (step 1).
        # Defer duplicate/length checks to the LLM (step 2) or the fallback
        # heuristic (step 3) so the LLM gets the first opportunity to validate
        # completeness and distinctness.
        resp1, resp2 = valid_responses[0], valid_responses[1]

        # Step 2 / 3: use LLM if available, otherwise fallback heuristic. The
        # helper will try LLM first and only run the heuristic if the LLM call
        # fails or the client is not present.
        return self._check_valid_two_responses_with_reason(resp1, resp2)

    def _check_valid_two_responses_with_reason(self, resp1: str, resp2: str) -> tuple:
        """Check if two responses are complete, standalone, and semantically different.

        Args:
          resp1: First response
          resp2: Second response

        Returns:
          Tuple of (success: bool, reason: str)
        """
        # If no OpenAI client provided, use heuristic
        if self.openai_client is None:
            success = self._check_semantic_difference_heuristic(resp1, resp2)
            if success:
                resp1_preview = resp1[:80].replace("\n", " ")
                resp2_preview = resp2[:80].replace("\n", " ")
                return (
                    True,
                    f"Instruction followed: Two distinct responses properly separated (fallback heuristic). Response 1: '{resp1_preview}...' Response 2: '{resp2_preview}...'",
                )
            else:
                return (
                    False,
                    "Instruction not followed: Two Responses are present but not semantically different (fallback heuristic)",
                )

        try:
            # Single LLM call to validate all criteria and get detailed reason
            # Focus the LLM only on completeness and exact-duplication (not semantic
            # diversity). The user's requirement: two distinct responses (not exact
            # duplicates), each complete and standalone. Do NOT penalize partial
            # semantic overlap.
            prompt = f"""Evaluate the following responses:

Response 1:
{resp1[:1000]}

Response 2:
{resp2[:1000]}

Answer ONLY in JSON with these keys:
{{
  "valid": true or false,
  "reason": "brief explanation"
}}

Criteria (both must be true for valid=true):
1) Each response is a complete, standalone answer (not a fragment or continuation).
2) The two responses are not effectively identical. For this comparison, consider responses identical if they are the same after lowercasing, trimming whitespace, collapsing internal whitespace, and removing punctuation etc. Small edits or formatting differences should NOT make two otherwise-equal responses count as distinct.

Do NOT evaluate broader semantic diversity or which response is "better". If invalid, state which criterion failed (e.g., "Response 1 is incomplete", "Responses are effectively identical after normalization").

Respond with only valid JSON, no additional text.

Response:"""

            # Call OpenAI API
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=15,
                    max_tokens=200,
                )
            except Exception as api_err:
                raise RuntimeError(
                    f"OpenAI API call failed: {type(api_err).__name__}: {str(api_err)}"
                )

            # Extract and parse the result
            try:
                response_content = response.choices[0].message.content.strip()
                if not response_content:
                    raise ValueError("LLM returned empty response content")
            except (AttributeError, IndexError) as extract_err:
                raise RuntimeError(
                    f"Failed to extract response content: {type(extract_err).__name__}: {str(extract_err)}"
                )

            try:
                result = _parse_llm_json_response(response_content)
            except (json.JSONDecodeError, ValueError) as parse_err:
                raise RuntimeError(f"JSON parsing error: {str(parse_err)}")

            # Validate result structure
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"Expected JSON dict, got {type(result).__name__}: {str(result)[:100]}"
                )

            is_valid = result.get("valid", False)
            llm_reason = result.get("reason", "No reason provided")

            if is_valid:
                return True, f"Instruction followed (LLM): {llm_reason}"
            else:
                return False, f"Instruction not followed (LLM): {llm_reason}"

        except Exception as e:
            # If OpenAI call fails, fall back to heuristic
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"OpenAI semantic check failed ({type(e).__name__}): {str(e)}")

            success = self._check_semantic_difference_heuristic(resp1, resp2)
            if success:
                resp1_preview = resp1[:80].replace("\n", " ")
                resp2_preview = resp2[:80].replace("\n", " ")
                return (
                    True,
                    f"Instruction followed (heuristic): Two semantically distinct responses properly separated. Response 1: '{resp1_preview}...' Response 2: '{resp2_preview}...'",
                )
            else:
                return (
                    False,
                    "Instruction not followed (heuristic): Responses are present but not semantically different",
                )

    def _check_semantic_difference_heuristic(self, resp1: str, resp2: str) -> bool:
        """Fallback heuristic to check if responses are semantically different.

        Args:
          resp1: First response
          resp2: Second response

        Returns:
          True if responses appear semantically different based on word overlap
        """

        # Normalize responses the same way we use above for duplicate detection
        def _normalize_for_dup(s: str) -> str:
            s2 = s.strip().lower()
            s2 = re.sub(r"\s+", " ", s2)
            s2 = re.sub(r"[^\w\s]", "", s2)
            return s2

        n1 = _normalize_for_dup(resp1)
        n2 = _normalize_for_dup(resp2)

        # If either normalized response is empty, treat as invalid — we do not
        # accept empty responses. This is stricter than before: return False to
        # indicate they do not meet the requirement for two distinct, complete
        # responses.
        if not n1 or not n2:
            return False

        # Enforce minimum word count per response (at least 5 words each).
        if len(resp1.split()) < 5 or len(resp2.split()) < 5:
            return False

        # If normalized strings are identical, they're effectively duplicates
        if n1 == n2:
            return False

        # Compute Jaccard similarity on normalized tokens as a fallback;
        # use a high threshold to only flag near-duplicates (e.g., >90%).
        words1 = set(n1.split())
        words2 = set(n2.split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0

        # Consider them duplicates if similarity is extremely high
        if similarity >= 0.90:
            return False

        # Otherwise, consider them sufficiently different
        return True
