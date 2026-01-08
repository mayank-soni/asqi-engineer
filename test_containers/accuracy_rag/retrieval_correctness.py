from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference
import asyncio
from dataclasses import dataclass
import typing as t
import numpy as np
from langchain_core.callbacks import Callbacks
from ragas.metrics import NonLLMContextRecall


@dataclass
class MyNonLLMContextPrecisionWithReferenceAndExplanation(
    NonLLMContextPrecisionWithReference
):
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        explanation = {"context_comparison_pairs": []}
        for rc in retrieved_contexts:
            pair_scores = []
            for ref in reference_contexts:
                match_score = await self.distance_measure.single_turn_ascore(
                    SingleTurnSample(reference=rc, response=ref), callbacks
                )
                pair_scores.append(match_score)
                explanation["context_comparison_pairs"].append(
                    {
                        "retrieved_context": rc,
                        "reference_context": ref,
                        "match_score": match_score,
                        "distance_measure": self.distance_measure.distance_measure.name,
                    }
                )
            scores.append(max(pair_scores))
        scores = [1 if score >= self.threshold else 0 for score in scores]
        explanation["threshold"] = self.threshold
        explanation["verdict_list"] = scores
        final_score = self._calculate_average_precision(scores)
        explanation["explanation"] = (
            f"{sum(scores)} out of {len(scores)} retrieved contexts were relevant "
            f"(max similarity >= {self.threshold}). The average precision score of {final_score} "
            f"accounts for ranking, rewarding relevant contexts that appear earlier in the list."
        )
        return {"score": final_score, "explanation": explanation}

    def _calculate_average_precision(self, verdict_list: t.List[int]) -> float:
        score = np.nan

        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        return score


@dataclass
class MyNonLLMContextRecallWithReferenceAndExplanation(NonLLMContextRecall):
    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        retrieved_contexts = sample.retrieved_contexts
        reference_contexts = sample.reference_contexts
        assert retrieved_contexts is not None, "retrieved_contexts is empty"
        assert reference_contexts is not None, "reference_contexts is empty"

        scores = []
        explanation = {"context_comparison_pairs": []}
        for ref in reference_contexts:
            pair_scores = []
            for rc in retrieved_contexts:
                match_score = await self.distance_measure.single_turn_ascore(
                    SingleTurnSample(reference=rc, response=ref), callbacks
                )
                pair_scores.append(match_score)
                explanation["context_comparison_pairs"].append(
                    {
                        "reference_context": ref,
                        "retrieved_context": rc,
                        "match_score": match_score,
                        "distance_measure": self.distance_measure.distance_measure.name,
                    }
                )
            scores.append(max(pair_scores))
        verdict_list = [1 if score > self.threshold else 0 for score in scores]
        explanation["threshold"] = self.threshold
        explanation["verdict_list"] = verdict_list
        score = self._compute_score(scores)
        explanation["explanation"] = (
            f"{sum(verdict_list)} out of {len(verdict_list)} reference contexts were matched "
            f"(max similarity > {self.threshold}), leading to a score of {score}."
        )
        return {"score": score, "explanation": explanation}

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)

    def _compute_score(self, verdict_list: t.List[float]) -> float:
        response = [1 if score > self.threshold else 0 for score in verdict_list]
        denom = len(response)
        numerator = sum(response)
        score = numerator / denom if denom > 0 else np.nan
        return score


class RetrievalCorrectness:
    """Weighted retrieval correctness combining ContextPrecision and ContextRecall.

    Score = 0.5 * ContextPrecision + 0.5 * ContextRecall

    Returns a dict with:
    - score: weighted combination of precision and recall
    - explanation_metadata: contains both metrics' explanations
    - explanation_summary: human-readable summary of the scores
    """

    def __init__(self, weight_precision: float = 0.5, weight_recall: float = 0.5):
        """Initialize with weights for precision and recall.

        Args:
            weight_precision: Weight for context precision (default: 0.5)
            weight_recall: Weight for context recall (default: 0.5)
        """
        self.weight_precision = weight_precision
        self.weight_recall = weight_recall
        self.precision_metric = MyNonLLMContextPrecisionWithReferenceAndExplanation()
        self.recall_metric = MyNonLLMContextRecallWithReferenceAndExplanation()

    async def ascore(self, sample: SingleTurnSample) -> dict:
        """Calculate weighted retrieval correctness score.

        Args:
            sample: SingleTurnSample with retrieved_contexts and reference_contexts

        Returns:
            Dict with score, explanation_metadata, and explanation_summary
        """
        # Calculate precision
        precision_result = await self.precision_metric._single_turn_ascore(
            sample, callbacks=None
        )
        precision_score = precision_result.get("score", 0.0)
        precision_explanation = precision_result.get("explanation", {})

        # Calculate recall
        recall_result = await self.recall_metric._single_turn_ascore(
            sample, callbacks=None
        )
        recall_score = recall_result.get("score", 0.0)
        recall_explanation = recall_result.get("explanation", {})

        # Calculate weighted score
        weighted_score = (
            self.weight_precision * precision_score + self.weight_recall * recall_score
        )

        # Generate summary based on scores
        precision_str = f"{precision_score:.2f}"
        recall_str = f"{recall_score:.2f}"

        summary = f"Average of Context precision: {precision_str} and Context recall: {recall_str}."

        return {
            "score": float(weighted_score),
            "explanation_metadata": {
                "contextprecision_score": float(precision_score),
                "contextprecision_metadata": precision_explanation,
                "contextrecall_score": float(recall_score),
                "contextrecall_metadata": recall_explanation,
                "weights": {
                    "precision": self.weight_precision,
                    "recall": self.weight_recall,
                },
            },
            "explanation_summary": summary,
        }

    def score(self, sample: SingleTurnSample) -> dict:
        """Synchronous wrapper for ascore using asyncio.run()."""
        return asyncio.run(self.ascore(sample))
