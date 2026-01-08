from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from typing import Optional, Union
from dotenv import load_dotenv
from enum import Enum
import json

load_dotenv()


class RelevancyCategory(Enum):
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"

    @property
    def display_name(self) -> str:
        """Return human-readable name for the category"""
        mapping = {
            "relevant": "Relevant",
            "partially_relevant": "Partially Relevant",
            "not_relevant": "Not Relevant",
        }
        return mapping[self.value]


class CompletenessCategory(Enum):
    COMPLETE = "complete"
    PARTIALLY_COMPLETE = "partially_complete"
    INCOMPLETE = "incomplete"

    @property
    def display_name(self) -> str:
        """Return human-readable name for the category"""
        mapping = {
            "complete": "Complete",
            "partially_complete": "Partially Complete",
            "incomplete": "Incomplete",
        }
        return mapping[self.value]


class CustomRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.0,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.evaluation_model = evaluation_model
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    @property
    def __name__(self):
        return "Custom Relevance Metric"

    def measure(self, test_case: LLMTestCase):
        evaluation_prompt = f"""
Evaluate the relevance of the response to the input question. Consider how well the response addresses the user's needs and provides useful information.

Return only a JSON object:
{{
  "category": "relevant" | "partially_relevant" | "not_relevant",
  "reason": "<why>"
}}

Definitions and scoring:
- relevant (score: 1.0): High relevance - most/all statements are relevant to the question, provides a direct answer or concrete solution
- partially_relevant (score: 0.5): Partial relevance - only some statements are relevant, may provide helpful redirection or incomplete information
- not_relevant (score: 0.0): Low relevance - very little of the response relates to the question, unhelpful or off-topic

Examples:
- relevant: Direct answers like "You can return it in 30 days with receipt" / Direct Answer with relevant information
- partially_relevant: "I don't have that information, but let me connect you with customer service" / Incomplete but helpful responses
- not_relevant: "I don't know" / "The moon is bright tonight" (when asked about shoes) / Completely off-topic responses

Input: {test_case.input}
Answer: {test_case.actual_output}
"""

        result = self.evaluation_model.generate(evaluation_prompt)

        # Handle both string and tuple returns
        if isinstance(result, tuple):
            result = result[0] if result else ""
        elif not isinstance(result, str):
            result = str(result)

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            parsed = {"category": "not_relevant", "reason": "Failed to parse response"}

        category = parsed["category"]

        # Map to enum for validation and display
        try:
            relevancy_category = RelevancyCategory(category)
            category_display = relevancy_category.display_name
        except ValueError:
            # Fallback for invalid categories
            relevancy_category = RelevancyCategory.NOT_RELEVANT
            category_display = "Unknown Category"

        # Use enum for consistent scoring logic
        if relevancy_category == RelevancyCategory.RELEVANT:
            self.score = 1.0
        elif relevancy_category == RelevancyCategory.PARTIALLY_RELEVANT:
            self.score = 0.5
        else:  # not_relevant
            self.score = 0.0

        self.reason = f"Category: {category_display}. {parsed.get('reason', '')}"
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)


class CustomCompletenessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.0,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.evaluation_model = evaluation_model
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    @property
    def __name__(self):
        return "Custom Completeness Metric"

    def measure(self, test_case: LLMTestCase):
        evaluation_prompt = f"""
Evaluate how completely the response answers the input question. Consider whether all aspects of the question are addressed and if sufficient detail is provided.

Return only a JSON object:
{{
  "category": "complete" | "partially_complete" | "incomplete",
  "reason": "<why>"
}}

Definitions and scoring:
- complete (score: 1.0): Comprehensive answer - addresses all aspects of the question with sufficient detail, answers every part of the input
- partially_complete (score: 0.5): Partial answer - addresses some aspects but missing key elements, incomplete reasoning, or insufficient detail
- incomplete (score: 0.0): Inadequate answer - fails to address the main question, major gaps in information, or completely off-topic

Examples:
- complete: "You can return shoes within 30 days with receipt for full refund, or exchange for different size if available" (addresses both return and exchange options)
- partially_complete: "You can return it in 30 days" (addresses return but missing exchange option and details)
- incomplete: "I don't know" / "Maybe ask someone else" / responses that don't attempt to answer the question

Input: {test_case.input}
Answer: {test_case.actual_output}
"""

        result = self.evaluation_model.generate(evaluation_prompt)

        # Handle both string and tuple returns
        if isinstance(result, tuple):
            result = result[0] if result else ""
        elif not isinstance(result, str):
            result = str(result)

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            parsed = {"category": "incomplete", "reason": "Failed to parse response"}

        category = parsed["category"]

        # Map to enum for validation and display
        try:
            completeness_category = CompletenessCategory(category)
            category_display = completeness_category.display_name
        except ValueError:
            # Fallback for invalid categories
            completeness_category = CompletenessCategory.INCOMPLETE
            category_display = "Unknown Category"

        # Use enum for consistent scoring logic
        if completeness_category == CompletenessCategory.COMPLETE:
            self.score = 1.0
        elif completeness_category == CompletenessCategory.PARTIALLY_COMPLETE:
            self.score = 0.5
        else:  # incomplete
            self.score = 0.0

        self.reason = f"Category: {category_display}. {parsed.get('reason', '')}"
        self.success = True
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)


class HelpfulnessMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        include_scores: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.include_scores = include_scores

    def measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, completeness_metric = self.initialize_metrics()
            # Remember, deepeval's default metrics follow the same pattern as your custom metric!
            relevancy_metric.measure(test_case)
            completeness_metric.measure(test_case)

            # Custom logic to set score, reason, and success
            self.set_score_reason_success(relevancy_metric, completeness_metric)
            return self.score
        except Exception as e:
            # Set and re-raise error
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, completeness_metric = self.initialize_metrics()
            # Here, we use the a_measure() method instead so both metrics can run concurrently
            await relevancy_metric.a_measure(test_case)
            await completeness_metric.a_measure(test_case)

            # Custom logic to set score, reason, and success
            self.set_score_reason_success(relevancy_metric, completeness_metric)
            return self.score
        except Exception as e:
            # Set and re-raise error
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Helpfulness Metric (Composite Relevancy Completeness Metric)"

    ######################
    ### Helper methods ###
    ######################
    def initialize_metrics(self):
        relevancy_metric = CustomRelevancyMetric(
            evaluation_model=self.evaluation_model,
            include_reason=True,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
        )

        completeness_metric = CustomCompletenessMetric(
            evaluation_model=self.evaluation_model,
            include_reason=True,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
        )

        return relevancy_metric, completeness_metric

    def set_score_reason_success(
        self, relevancy_metric: BaseMetric, completeness_metric: BaseMetric
    ):
        # Get scores and reasons for both
        relevancy_score = relevancy_metric.score
        relevancy_reason = relevancy_metric.reason
        completeness_score = completeness_metric.score
        completeness_reason = completeness_metric.reason

        if self.include_scores:
            self.individual_scores = {
                "relevancy_score": relevancy_score,
                "completeness_score": completeness_score,
            }
        # Custom logic to set score
        composite_score = 0.5 * relevancy_score + 0.5 * completeness_score
        self.score = (
            0
            if self.strict_mode and composite_score < self.threshold
            else composite_score
        )

        # Custom logic to set reason
        if self.include_reason:
            self.reason = {
                "relevancy_reason": relevancy_reason,
                "completeness_reason": completeness_reason,
            }

        # Generate simplified summary based on individual scores
        relevancy_status = (
            "relevant"
            if relevancy_score >= 0.7
            else ("partially relevant" if relevancy_score >= 0.4 else "not relevant")
        )
        completeness_status = (
            "complete"
            if completeness_score >= 0.7
            else ("partially complete" if completeness_score >= 0.4 else "incomplete")
        )

        summary = f"Response is {relevancy_status} and {completeness_status}."

        # Store metadata and summary for later retrieval
        self.explanation_metadata = (
            self.reason if isinstance(self.reason, dict) else {"reason": self.reason}
        )
        self.explanation_summary = summary

        # Custom logic to set success
        self.success = self.score >= self.threshold
