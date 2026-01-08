from __future__ import annotations
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness

import typing as t
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ragas.metrics.utils import fbeta_score
from ragas.metrics.base import MetricType

try:
    from .faithfulness import MyNLIStatementInput
except ImportError:
    from faithfulness import MyNLIStatementInput

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.dataset_schema import SingleTurnSample


@dataclass
class MyFactualCorrectnessWithExplanation(FactualCorrectness):
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response", "reference", "user_input"}
        }
    )

    async def verify_claims(
        self,
        premise: str,
        user_instruction: str,
        hypothesis_list: t.List[str],
        callbacks: Callbacks,
    ) -> NDArray[np.bool_]:
        assert self.llm is not None, "LLM must be set"
        prompt_input = MyNLIStatementInput(
            context=premise,
            user_instruction=user_instruction,
            statements=hypothesis_list,
        )
        response = await self.nli_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        if response.statements:
            claim_verifications = np.array(
                [bool(result.verdict) for result in response.statements]
            )
            claim_verifications_with_reasons = [
                {
                    "statement": result.statement,
                    "reason": result.reason,
                    "verdict": result.verdict,
                }
                for result in response.statements
            ]
        else:
            claim_verifications = np.array([], dtype=bool)
            claim_verifications_with_reasons = []
        return claim_verifications, claim_verifications_with_reasons

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        reference = sample.reference
        response = sample.response
        user_instruction = sample.user_input
        assert self.llm is not None, "LLM must be set"
        assert reference is not None, "Reference is not set"
        assert response is not None, "Response is not set"
        assert user_instruction is not None, "User instruction is not set"

        explanation = {}

        response_claims = await self.decompose_claims(response, callbacks)
        reference_response, reference_response_with_reasons = await self.verify_claims(
            premise=reference,
            user_instruction=user_instruction,
            hypothesis_list=response_claims,
            callbacks=callbacks,
        )

        explanation["response_claims_verification_with_reference_answer"] = (
            reference_response_with_reasons
        )

        if self.mode != "precision":
            reference_claims = await self.decompose_claims(reference, callbacks)
            (
                response_reference,
                response_reference_with_reasons,
            ) = await self.verify_claims(
                premise=response,
                user_instruction=user_instruction,
                hypothesis_list=reference_claims,
                callbacks=callbacks,
            )
            explanation["reference_answer_claims_verification_with_response"] = (
                response_reference_with_reasons
            )
        else:
            response_reference = np.array([], dtype=bool)

        tp = sum(reference_response)
        fp = sum(~reference_response)
        if self.mode != "precision":
            fn = sum(~response_reference)
        else:
            fn = 0

        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)

        explanation["mode"] = self.mode
        explanation["tp"] = int(tp)
        explanation["fp"] = int(fp)
        explanation["fn"] = int(fn)
        explanation["explanation"] = (
            f"Out of {len(response_claims)} response claims, {tp} were verified as correct against the reference. "
            f"See detailed verifications for breakdowns."
        )

        # Generate simplified summary explanation
        score_rounded = np.round(score, 2)
        if score_rounded >= 0.7:
            summary = "The model's answer matches the reference answer well."
        elif score_rounded >= 0.4:
            summary = "The model's answer partially matches the reference answer."
        else:
            summary = "The model's answer does not match the reference answer well."
        summary += (
            f" {int(tp)} out of {len(response_claims)} claims were verified as correct."
        )

        return {
            "score": np.round(score, 2),
            "explanation_metadata": explanation,
            "explanation_summary": summary,
        }
