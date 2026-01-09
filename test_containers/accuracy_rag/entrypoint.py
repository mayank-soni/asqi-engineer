import argparse
import asyncio
import csv
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from answer_correctness import MyFactualCorrectnessWithExplanation
from custom_gpt_model import CustomGPTModel
from deepeval.test_case import LLMTestCase
from faithfulness import MyFaithfulnessWithInstructions
from helpfulness import HelpfulnessMetric
from if_metric import InstructionFollowingMetric
from input_schemas import DatasetValidator
from openai import AsyncOpenAI, OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from relaxed_retrieval_correctness import RelaxedRetrievalCorrectness
from retrieval_correctness import RetrievalCorrectness

from asqi.datasets import Dataset, load_hf_dataset

os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"
os.environ["CONFIDENT_METRIC_LOGGING_FLUSH"] = "0"
logging.getLogger("deepeval").setLevel(logging.ERROR)
logging.getLogger("ragas").setLevel(logging.ERROR)

RESULTS_DIR = "/output/accuracy_rag_results"
RAG_ACCURACY_DATASET_NAME = "rag_accuracy_dataset"
RAG_IF_DATASET_NAME = "rag_if_dataset"

def humanize_instruction_type(name: str) -> str:
    """Convert snake_case instruction type to a human-friendly title.

    Example: 'two_responses' -> 'Two Responses'
    """
    if not name or not isinstance(name, str):
        return name
    parts = [p for p in name.split("_") if p]
    return " ".join(p.capitalize() for p in parts)


class NonIFAccuracyTestRunner:
    """Runs accuracy evaluation on LLM responses with multiple metrics"""

    def __init__(self, systems_params: Dict[str, Any], test_params: Dict[str, Any]):
        self.systems_params = systems_params
        self.test_params = test_params

        sut_params = systems_params.get("system_under_test", {})
        self.sut_model = sut_params.get("model")
        self.sut_base_url = sut_params.get("base_url")
        self.sut_api_key = sut_params.get("api_key")

        self.evaluator_llm = self._setup_evaluator_llm("evaluator_system")
        self.evaluator_model = self._setup_evaluator_model("evaluator_system")
        self.sut_client = AsyncOpenAI(
            api_key=self.sut_api_key, base_url=self.sut_base_url
        )

        self.results = []
        self.metric_scores = {
            "conditional_task_success": [],
            "answer_correctness": [],
            "faithfulness": [],
            "retrieval_correctness": [],
            "helpfulness": [],
            "hit@k": [],
            "retrieval_hard_gate": [],
            "faithfulness_hard_gate": [],
            "hard_gate_pass": [],
        }

    def _setup_evaluator_llm(self, system_role: str) -> Optional[Any]:
        """Setup evaluator LLM using llm_factory with fallback to OpenAI"""
        system_config = self.systems_params.get(system_role)

        # If system_config is empty or not provided, use OpenAI fallback
        if not system_config or not isinstance(system_config, dict):
            system_config = {}

        api_key = system_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = system_config.get("base_url") or "https://api.openai.com/v1"
        model = system_config.get("model", "gpt-4o-mini")

        if not api_key:
            return None

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            return llm_factory(model, client=client)
        except Exception as e:
            print(f"ERROR: Failed to setup evaluator LLM: {str(e)}", file=sys.stderr)
            return None

    def _setup_evaluator_model(self, system_role: str) -> Optional[CustomGPTModel]:
        """Setup evaluator model for deepeval with fallback to OpenAI"""
        system_config = self.systems_params.get(system_role)

        # If system_config is empty or not provided, use OpenAI fallback
        if not system_config or not isinstance(system_config, dict):
            system_config = {}

        api_key = system_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = system_config.get("base_url") or "https://api.openai.com/v1"
        model = system_config.get("model", "gpt-4o-mini")

        if not api_key:
            return None

        return CustomGPTModel(model=model, _openai_api_key=api_key, base_url=base_url)

    async def load_questions(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """Load dataset rows from JSONL file with validation"""
        # Validate input file schema first
        validation_result = DatasetValidator.validate_rag_accuracy_hf(dataset)
        if not validation_result["valid"]:
            error_msg = f"Invalid RAG accuracy dataset: {', '.join(validation_result['errors'][:5])}"
            if len(validation_result["errors"]) > 5:
                error_msg += (
                    f" (and {len(validation_result['errors']) - 5} more errors)"
                )
            raise ValueError(error_msg)

        rows: List[Dict[str, Any]] = []
        try:
            for row in dataset:
                question = row.get("question", "")
                answer = row.get("answer", "")
                context = row.get("context", [])

                if isinstance(context, str):
                    context = [context]

                rows.append(
                    {
                        "question": question.strip()
                        if isinstance(question, str)
                        else question,
                        "answer": answer.strip() if isinstance(answer, str) else answer,
                        "context": context,
                    }
                )
            return rows
        except Exception as e:
            raise ValueError(f"Failed to load dataset from JSONL: {e}")

    async def get_sut_response(self, question: str) -> tuple[str, List[str]]:
        """Get response from system under test. Returns (answer, retrieved_contexts)"""
        try:
            response = await self.sut_client.chat.completions.create(
                model=self.sut_model,
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content
            retrieved_contexts = []
            message_dict = (
                response.choices[0].message.model_dump()
                if hasattr(response.choices[0].message, "model_dump")
                else response.choices[0].message.__dict__
            )

            if "context" in message_dict:
                context = message_dict.get("context")
                if isinstance(context, list):
                    retrieved_contexts = [str(ctx) for ctx in context if ctx]
                elif isinstance(context, dict) and "citations" in context:
                    for citation in context.get("citations", []):
                        if isinstance(citation, dict):
                            retrieved_contexts.append(
                                citation.get("retrieved_context", "")
                            )
                        elif isinstance(citation, str):
                            retrieved_contexts.append(citation)

            return answer, retrieved_contexts
        except Exception as e:
            print(f"Error generating response: {str(e)}", file=sys.stderr)
            return f"Error: {str(e)}", []

    async def evaluate_answer_correctness(
        self, question: str, answer: str, reference_answer: str = None
    ) -> Dict[str, Any]:
        """Evaluate answer correctness using RAGAS metric"""
        try:
            sample = SingleTurnSample(
                user_input=question, response=answer, reference=reference_answer
            )
            metric = MyFactualCorrectnessWithExplanation(
                llm=self.evaluator_llm, mode="recall"
            )
            result = await metric.single_turn_ascore(sample)
            if isinstance(result, dict):
                return result
            else:
                score = result if isinstance(result, (int, float)) else 0.0
                return {
                    "score": float(score),
                    "explanation_metadata": {},
                    "explanation_summary": "",
                }
        except Exception as e:
            print(f"Error in answer correctness evaluation: {e}", file=sys.stderr)
            return {
                "score": 0.0,
                "explanation_metadata": {},
                "explanation_summary": f"Error: {str(e)}",
            }

    async def evaluate_faithfulness(
        self, question: str, answer: str, retrieved_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate faithfulness of answer to context"""
        try:
            contexts = retrieved_contexts or [question]
            sample = SingleTurnSample(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            metric = MyFaithfulnessWithInstructions(llm=self.evaluator_llm)
            result = await metric.single_turn_ascore(sample)
            if isinstance(result, dict):
                return result
            else:
                score = result if isinstance(result, (int, float)) else 0.0
                return {
                    "score": float(score),
                    "explanation_metadata": {},
                    "explanation_summary": "",
                }
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}", file=sys.stderr)
            return {
                "score": 0.0,
                "explanation_metadata": {},
                "explanation_summary": f"Error: {str(e)}",
            }

    async def evaluate_retrieval_correctness(
        self,
        question: str,
        retrieved_contexts: List[str] = None,
        reference_contexts: List[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate retrieval correctness"""
        try:
            r_contexts = retrieved_contexts or [question]
            ref_contexts = reference_contexts or [question]

            sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=r_contexts,
                reference_contexts=ref_contexts,
            )
            metric = RetrievalCorrectness()
            result = await metric.ascore(sample)

            # Result is now a dict with score, explanation_metadata, and explanation_summary
            if isinstance(result, dict):
                return result
            else:
                return {
                    "score": 0.0,
                    "explanation_metadata": {},
                    "explanation_summary": "Error in retrieval correctness evaluation",
                }
        except Exception as e:
            print(f"Error in retrieval correctness evaluation: {e}", file=sys.stderr)
            return {
                "score": 0.0,
                "explanation_metadata": {},
                "explanation_summary": f"Error: {str(e)}",
            }

    async def evaluate_relaxed_retrieval_correctness(
        self,
        question: str,
        retrieved_contexts: List[str] = None,
        reference_contexts: List[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate retrieval correctness"""
        try:
            r_contexts = retrieved_contexts or [question]
            ref_contexts = reference_contexts or [question]

            metric = RelaxedRetrievalCorrectness()
            k = 5
            result = metric.hit_at_k(r_contexts, ref_contexts, k=k)

            # Normalize/augment result to ensure threshold and metric info present
            if isinstance(result, dict):
                meta = (
                    result.get("explanation_metadata", {})
                    if isinstance(result.get("explanation_metadata", {}), dict)
                    else {}
                )
                meta.update({"k": k})
                # convert hit indication
                hit = bool(result.get("score", 0.0) >= 1.0)
                meta.setdefault("hit", hit)
                result["explanation_metadata"] = meta
                # ensure one-line summary
                if not result.get("explanation_summary"):
                    result["explanation_summary"] = (
                        "Reference found in top-{k}".format(k=k)
                        if hit
                        else f"No reference found in top-{k}"
                    )
                return result
            else:
                return {
                    "score": 0.0,
                    "explanation_metadata": {"k": k, "hit": False},
                    "explanation_summary": f"Error in relaxed retrieval correctness evaluation for top-{k}",
                }
        except Exception as e:
            print(f"Error in retrieval correctness evaluation: {e}", file=sys.stderr)
            return {
                "score": 0.0,
                "explanation_metadata": {},
                "explanation_summary": f"Error: {str(e)}",
            }

    async def evaluate_helpfulness(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate helpfulness of the answer"""
        try:
            test_case = LLMTestCase(input=question, actual_output=answer)
            metric = HelpfulnessMetric(
                threshold=0.5,
                evaluation_model=self.evaluator_model,
                include_reason=True,
                async_mode=True,
                strict_mode=False,
                include_scores=True,
            )
            await metric.a_measure(test_case)
            return {
                "score": metric.score,
                "explanation_metadata": metric.explanation_metadata
                if hasattr(metric, "explanation_metadata")
                else metric.reason
                if hasattr(metric, "reason")
                else {},
                "explanation_summary": metric.explanation_summary
                if hasattr(metric, "explanation_summary")
                else "",
            }
        except Exception as e:
            print(f"Error in helpfulness evaluation: {e}", file=sys.stderr)
            return {
                "score": 0.0,
                "explanation_metadata": {},
                "explanation_summary": f"Error: {str(e)}",
            }

    async def run_evaluation(
        self, dataset: Dataset, max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run full evaluation on accuracy dataset"""
        try:
            rows = await self.load_questions(dataset)
            if not rows:
                return {
                    "success": False,
                    "error": "No rows found in dataset",
                    "conditional_task_sucess": 0.0,
                    "answer_correctness": 0.0,
                    "faithfulness": 0.0,
                    "retrieval_correctness": 0.0,
                    "helpfulness": 0.0,
                    "questions_evaluated": 0,
                    "details": [],
                }

            for idx, row in enumerate(rows, 1):
                if max_samples and idx > max_samples:
                    break

                question = row.get("question", "")
                answer = row.get("answer", "")
                context = row.get("context", [])
                sut_answer, retrieved_contexts = await self.get_sut_response(question)
                reference_contexts = context if context else [question]

                ac_result = await self.evaluate_answer_correctness(
                    question, sut_answer, reference_answer=answer
                )
                faith_result = await self.evaluate_faithfulness(
                    question, sut_answer, retrieved_contexts=retrieved_contexts
                )
                faith_score = faith_result.get("score", 0.0)
                retrieval_result = await self.evaluate_retrieval_correctness(
                    question,
                    retrieved_contexts=retrieved_contexts,
                    reference_contexts=reference_contexts,
                )
                relaxed_retrieval_result = (
                    await self.evaluate_relaxed_retrieval_correctness(
                        question,
                        retrieved_contexts=retrieved_contexts,
                        reference_contexts=reference_contexts,
                    )
                )
                # Derived metric: retrieval grounding (pass if relaxed_retrieval hit@k == 1)
                try:
                    relaxed_score = float(relaxed_retrieval_result.get("score", 0.0))
                except Exception:
                    relaxed_score = 0.0
                grounding_pass = 1.0 if relaxed_score >= 1.0 else 0.0
                retrieval_hard_gate = {
                    "score": float(grounding_pass),
                    "explanation_metadata": {
                        "relaxed_retrieval_score": relaxed_score,
                        "threshold": 1.0,
                        "k": relaxed_retrieval_result.get(
                            "explanation_metadata", {}
                        ).get("k")
                        if isinstance(
                            relaxed_retrieval_result.get("explanation_metadata", {}),
                            dict,
                        )
                        else None,
                    },
                    "explanation_summary": "Reference present within top-k retrieved contexts"
                    if grounding_pass == 1.0
                    else "No reference within top-k retrieved contexts",
                }

                # Derived metric: faithfulness hard gate (pass if faithfulness >= 0.70)
                faith_score_val = float(faith_score or 0.0)
                faith_gate_pass = 1.0 if faith_score_val >= 0.70 else 0.0
                faithfulness_hard_gate = {
                    "score": float(faith_gate_pass),
                    "explanation_metadata": {
                        "faithfulness_score": faith_score_val,
                        "threshold": 0.70,
                    },
                    "explanation_summary": "Faithfulness meets threshold (>= 0.70)"
                    if faith_gate_pass == 1.0
                    else "Faithfulness below threshold (< 0.70)",
                }
                help_result = await self.evaluate_helpfulness(question, sut_answer)

                self.metric_scores["answer_correctness"].append(
                    ac_result.get("score", 0.0)
                )
                self.metric_scores["faithfulness"].append(
                    faith_result.get("score", 0.0)
                )
                self.metric_scores["retrieval_correctness"].append(
                    retrieval_result.get("score", 0.0)
                )
                self.metric_scores["hit@k"].append(
                    relaxed_retrieval_result.get("score", 0.0)
                )
                self.metric_scores["retrieval_hard_gate"].append(
                    retrieval_hard_gate.get("score", 0.0)
                )
                self.metric_scores["faithfulness_hard_gate"].append(
                    faithfulness_hard_gate.get("score", 0.0)
                )
                self.metric_scores["helpfulness"].append(help_result.get("score", 0.0))

                # Evaluate sample-level hard gates and compute task success
                failed_gates: List[str] = []
                # retrieval_hard_gate and faithfulness_hard_gate are applicable
                if retrieval_hard_gate.get("score", 0.0) < 1.0:
                    failed_gates.append("retrieval_hard_gate")
                if faithfulness_hard_gate.get("score", 0.0) < 1.0:
                    failed_gates.append("faithfulness_hard_gate")

                task_success_score = 0.0

                if failed_gates:
                    # Any failed hard gate → immediate failure
                    hard_gate_pass = 0.0
                    task_success_explanation = {
                        "failed_gates": failed_gates,
                        "retrieval_hard_gate": retrieval_hard_gate,
                        "faithfulness_hard_gate": faithfulness_hard_gate,
                        "hard_gate_pass": hard_gate_pass,
                    }
                    task_success_summary = (
                        f"Failed hard gates: {', '.join(failed_gates)}"
                    )
                else:
                    # All hard gates passed → compute continuous task success
                    hard_gate_pass = 1.0
                    ac_score_val = float(ac_result.get("score", 0.0))
                    help_score_val = float(help_result.get("score", 0.0))
                    # Continuous task success formula (changeable)
                    task_success_score = 0.7 * ac_score_val + 0.3 * help_score_val
                    task_success_explanation = {
                        "hard_gate_pass": hard_gate_pass,
                        "answer_correctness": ac_score_val,
                        "helpfulness": help_score_val,
                        "formula": "0.70*answer_correctness + 0.30*helpfulness",
                    }
                    task_success_summary = f"All hard gates passed. Task success computed as {task_success_score:.2f}"

                # record per-sample task success
                self.metric_scores["conditional_task_success"].append(float(task_success_score))
                # record hard_gate_pass per sample (1.0 if all hard gates passed, else 0.0)
                self.metric_scores["hard_gate_pass"].append(float(hard_gate_pass))

                task_success_output = {
                    "score": float(task_success_score),
                    "explanation_metadata": task_success_explanation,
                    "explanation_summary": task_success_summary,
                }
                # expose hard_gate_pass as a standardized metric as well
                hard_gate_pass_metric = {
                    "score": float(hard_gate_pass),
                    "explanation_metadata": {"failed_gates": failed_gates}
                    if failed_gates
                    else {"hard_gate_pass": hard_gate_pass},
                    "explanation_summary": "All hard gates passed"
                    if hard_gate_pass == 1.0
                    else f"Failed gates: {', '.join(failed_gates)}",
                }

                self.results.append(
                    {
                        "question_index": idx,
                        "question": question,
                        "sut_answer": sut_answer,
                        "reference_answer": answer,
                        "reference_contexts": context,
                        "retrieved_contexts": retrieved_contexts,
                        "metrics": {
                            "answer_correctness": ac_result,
                            "faithfulness": faith_result,
                            "helpfulness": help_result,
                            "retrieval_correctness": retrieval_result,
                            "hit@k": relaxed_retrieval_result,
                            "conditional_task_success": task_success_output,
                            "retrieval_hard_gate": retrieval_hard_gate,
                            "faithfulness_hard_gate": faithfulness_hard_gate,
                            "hard_gate_pass": hard_gate_pass_metric
                        },
                    }
                )

                print(
                    f"[{idx}/{len(rows)}] Question: {question[:50]}... | AC: {ac_result.get('score', 0.0):.2f} | Faith: {faith_result.get('score', 0.0):.2f}",
                    file=sys.stderr,
                )

            avg_correctness = (
                np.mean(self.metric_scores["answer_correctness"])
                if self.metric_scores["answer_correctness"]
                else 0.0
            )
            avg_faithfulness = (
                np.mean(self.metric_scores["faithfulness"])
                if self.metric_scores["faithfulness"]
                else 0.0
            )
            avg_retrieval = (
                np.mean(self.metric_scores["retrieval_correctness"])
                if self.metric_scores["retrieval_correctness"]
                else 0.0
            )
            avg_hit_at_k = (
                np.mean(self.metric_scores["hit@k"])
                if self.metric_scores["hit@k"]
                else 0.0
            )
            avg_retrieval_grounding = (
                np.mean(self.metric_scores["retrieval_hard_gate"])
                if self.metric_scores["retrieval_hard_gate"]
                else 0.0
            )
            avg_faithfulness_hard_gate = (
                np.mean(self.metric_scores["faithfulness_hard_gate"])
                if self.metric_scores["faithfulness_hard_gate"]
                else 0.0
            )
            avg_helpfulness = (
                np.mean(self.metric_scores["helpfulness"])
                if self.metric_scores["helpfulness"]
                else 0.0
            )
            avg_task_success = (
                np.mean(self.metric_scores["conditional_task_success"])
                if self.metric_scores["conditional_task_success"]
                else 0.0
            )

            # Additional aggregated failure metrics
            # Use length of metric_scores lists (number of samples actually processed) not len(rows)
            num_samples = (
                len(self.metric_scores["hard_gate_pass"])
                if self.metric_scores["hard_gate_pass"]
                else 0
            )
            

            # Compute S_base: mean(task_success_i | hard_gate_pass_i == 1)
            hard_gate_pass_vals = self.metric_scores.get("hard_gate_pass", [])
            task_success_vals = self.metric_scores.get("conditional_task_success", [])
            passed_count = 0
            passed_task_success_sum = 0.0
            if hard_gate_pass_vals and task_success_vals:
                for ts_val, gp_val in zip(task_success_vals, hard_gate_pass_vals):
                    try:
                        if float(gp_val) >= 1.0:
                            passed_count += 1
                            passed_task_success_sum += float(ts_val)
                    except Exception:
                        continue
            S_base = (passed_task_success_sum / passed_count) if passed_count else 0.0
            P_base_count = passed_count
            N_base = num_samples
            hard_gate_pass_rate = (P_base_count / N_base) if N_base else 0.0
            return {
                "success": True,
                "answer_correctness": float(avg_correctness),
                "faithfulness": float(avg_faithfulness),
                "helpfulness": float(avg_helpfulness),
                "retrieval_correctness": float(avg_retrieval),
                "hit@k": float(avg_hit_at_k),
                "conditional_task_success": float(avg_task_success),
                "retrieval_gate_pass_rate": float(avg_retrieval_grounding),
                "faithfulness_gate_pass_rate": float(avg_faithfulness_hard_gate),
                "hard_gate_pass_rate": float(hard_gate_pass_rate),
                "conditional_task_success": float(S_base),
                "questions_evaluated": len(rows),
                "details": self.results,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer_correctness": 0.0,
                "faithfulness": 0.0,
                "retrieval_correctness": 0.0,
                "conditional_task_success": 0.0,
                "questions_evaluated": 0,
            }


class IFTestRunner:
    """Runs instruction-following evaluation on LLM responses"""

    def __init__(self, systems_params: Dict[str, Any], test_params: Dict[str, Any]):
        sut_params = systems_params.get("system_under_test", {})
        self.sut_model = sut_params.get("model")
        self.sut_base_url = sut_params.get("base_url")
        self.sut_api_key = sut_params.get("api_key", os.getenv("OPENAI_API_KEY"))
        self.sut_client = AsyncOpenAI(
            api_key=self.sut_api_key, base_url=self.sut_base_url
        )

        evaluator_params = systems_params.get("evaluator_system", {})
        self.llm_model = (
            evaluator_params.get("model", "gpt-4o-mini")
            if evaluator_params
            else "gpt-4o-mini"
        )
        evaluator_api_key = evaluator_params.get("api_key", os.getenv("OPENAI_API_KEY"))
        evaluator_base_url = evaluator_params.get("base_url")
        self.evaluator_client = (
            OpenAI(api_key=evaluator_api_key, base_url=evaluator_base_url)
            if evaluator_api_key
            else None
        )

        self.metric = InstructionFollowingMetric(
            llm_model=self.llm_model, openai_client=self.evaluator_client
        )
        self.results = []
        self.metric_scores = {"instruction_following": []}

    async def load_questions_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load instruction-following dataset rows from CSV file"""
        # Validate input file schema first
        validation_result = DatasetValidator.validate_instruction_following_csv(
            csv_path
        )
        if not validation_result["valid"]:
            error_msg = f"Invalid instruction-following dataset: {', '.join(validation_result['errors'][:5])}"
            if len(validation_result["errors"]) > 5:
                error_msg += (
                    f" (and {len(validation_result['errors']) - 5} more errors)"
                )
            raise ValueError(error_msg)

        rows: List[Dict[str, Any]] = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    original_question = row.get("original_question", "").strip()
                    transformed_question = row.get("transformed_question", "").strip()
                    instruction_type = row.get("instruction_type", "").strip()
                    instruction_type_metadata = row.get(
                        "instruction_type_metadata", ""
                    ).strip()

                    if not transformed_question:
                        continue

                    metadata = {}
                    if instruction_type_metadata:
                        try:
                            metadata = json.loads(instruction_type_metadata)
                        except json.JSONDecodeError:
                            metadata = {"raw": instruction_type_metadata}

                    rows.append(
                        {
                            "question": transformed_question,
                            "original_question": original_question,
                            "instruction_type": instruction_type,
                            "instruction_type_metadata": metadata,
                        }
                    )
            return rows
        except Exception as e:
            raise ValueError(
                f"Failed to load instruction-following dataset from CSV: {e}"
            )

    async def get_sut_response(self, question: str) -> str:
        """Get response from system under test"""
        try:
            response = await self.sut_client.chat.completions.create(
                model=self.sut_model,
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}", file=sys.stderr)
            return f"Error: {str(e)}"

    async def run_evaluation(
        self, dataset_path: str, max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run instruction-following evaluation"""
        try:
            rows = await self.load_questions_from_csv(dataset_path)
            if not rows:
                return {
                    "success": False,
                    "error": "No rows found in dataset file",
                    "instruction_following_score": 0.0,
                    "questions_evaluated": 0,
                    "passing": 0,
                    "failing": 0,
                    "details": [],
                }

            if max_samples and max_samples < len(rows):
                rows = rows[:max_samples]

            passing = 0
            total_score = 0.0
            invalid_count = 0
            valid_count = 0

            for idx, row in enumerate(rows, 1):
                question = row.get("question", "")
                original_question = row.get("original_question", "")
                instruction_type = row.get("instruction_type", "")
                # keep the raw instruction_type for metric evaluation but present a humanized
                # version in logs and saved results (e.g. 'two_responses' -> 'Two Responses')
                instruction_type_raw = instruction_type
                instruction_type_display = humanize_instruction_type(
                    instruction_type_raw
                )
                instruction_type_metadata = row.get("instruction_type_metadata", {})

                answer = await self.get_sut_response(question)
                print(f"[IF] Question {idx}: {question[:50]}...", file=sys.stderr)
                print(
                    f"[IF] Instruction Type: {instruction_type_display}",
                    file=sys.stderr,
                )
                print(f"[IF] Answer: {answer[:100]}...", file=sys.stderr)

                instruction_metadata_str = (
                    json.dumps(instruction_type_metadata)
                    if isinstance(instruction_type_metadata, dict)
                    else str(instruction_type_metadata)
                )
                # pass the raw instruction_type to the metric implementation
                evaluation = self.metric.evaluate(
                    answer, instruction_type_raw, instruction_metadata_str
                )

                # Keep legacy fields for internal logic
                score = evaluation.get("score", 0.0)
                is_success = evaluation.get("success", False)

                # Aggregate metric score only for valid-format samples
                is_invalid = bool(evaluation.get("is_invalid_format", False))
                if is_invalid:
                    invalid_count += 1
                else:
                    # only include valid samples in per-metric aggregates
                    self.metric_scores["instruction_following"].append(score)
                    if is_success:
                        passing += 1
                    total_score += score
                    valid_count += 1

                # Normalize evaluation into the standardized three-field format
                reason = evaluation.get("reason", "")
                explanation_metadata = {
                    k: v
                    for k, v in evaluation.items()
                    if k not in ("score", "reason", "success")
                }

                standardized_evaluation = {
                    "score": score,
                    "explanation_metadata": explanation_metadata,
                    "explanation_summary": reason,
                }

                self.results.append(
                    {
                        "question_index": idx,
                        "question": question,
                        "sut_answer": answer,
                        "original_question": original_question,
                        # store the human-friendly label for inspection in results
                        "instruction_type": instruction_type_display,
                        "instruction_type_metadata": instruction_type_metadata,
                        "metrics": {
                            "instruction_following": standardized_evaluation,
                        },
                    }
                )

                status = "✓" if is_success else "✗"
                print(
                    f"[IF {idx}/{len(rows)}] {status} Score: {score:.2f}",
                    file=sys.stderr,
                )

            avg_score = (total_score / valid_count) if valid_count else 0.0
            failing = valid_count - passing

            return {
                "success": True,
                "instruction_following_score": float(avg_score),
                "questions_evaluated": len(rows),
                "valid_samples": int(valid_count),
                "invalid_samples": int(invalid_count),
                "passing": int(passing),
                "failing": int(failing),
                "pass_rate": (passing / valid_count * 100) if valid_count else 0.0,
                "details": self.results,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "instruction_following_score": 0.0,
                "questions_evaluated": 0,
                "passing": 0,
                "failing": 0,
                "pass_rate": 0.0,
                "details": [],
            }


def parse_arguments() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG accuracy evaluation test runner")
    parser.add_argument(
        "--systems-params",
        type=str,
        required=True,
        help="JSON string with system parameters",
    )
    parser.add_argument(
        "--test-params",
        type=str,
        required=True,
        help="JSON string with test parameters",
    )
    args = parser.parse_args()

    try:
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        return systems_params, test_params
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in arguments: {e}")


def validate_inputs(systems_params: Dict[str, Any], test_params: Dict[str, Any]):
    """Validate input parameters"""
    sut_params = systems_params.get("system_under_test", {})
    required_sut_fields = ["type", "base_url", "model"]
    for field in required_sut_fields:
        if field not in sut_params:
            raise ValueError(f"Missing required system_under_test parameter: {field}")

    if sut_params["type"] != "rag_api":
        raise ValueError(
            f"System under test must be 'rag_api', got: {sut_params['type']}"
        )


async def main_async():
    """Async main execution function"""
    try:
        systems_params, test_params = parse_arguments()
        validate_inputs(systems_params, test_params)

        max_rows = test_params.get("max_rows")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        accuracy_dataset_config = test_params.get("datasets")

        accuracy_dataset = load_hf_dataset(
            test_params["datasets"][RAG_ACCURACY_DATASET_NAME]
        )
        if_dataset = load_hf_dataset(test_params["datasets"][RAG_IF_DATASET_NAME])

        accuracy_runner = NonIFAccuracyTestRunner(systems_params, test_params)
        if_runner = IFTestRunner(systems_params, test_params)

        accuracy_result = await accuracy_runner.run_evaluation(
            accuracy_dataset, max_samples=max_rows
        )
        if_result = await if_runner.run_evaluation(if_dataset, max_samples=max_rows)

        detailed_results = {
            "base_accuracy_dataset": accuracy_result.get("details", []),
            "instruction_following_dataset": if_result.get("details", []),
        }

        with open(os.path.join(results_dir, "detailed_results.json"), "w") as f:
            json.dump(detailed_results, f, indent=2)

        retrieval_gate_pass_rate = float(accuracy_result.get("retrieval_gate_pass_rate", 0.0))

        faithfulness_gate_pass_rate = float(accuracy_result.get("faithfulness_gate_pass_rate", 0.0))

        hard_gate_pass_rate = float(accuracy_result.get("hard_gate_pass_rate", 0.0))
        conditional_task_success_val = float(
            accuracy_result.get("conditional_task_success", 0.0)
        )
        instr_following_score = float(if_result.get("instruction_following_score", 0.0))

        combined_result = {
            "success": accuracy_result.get("success") and if_result.get("success"),
            "conditional_task_success": accuracy_result.get("conditional_task_success", 0.0),
            "answer_correctness": accuracy_result.get("answer_correctness", 0.0),
            "faithfulness": accuracy_result.get("faithfulness", 0.0),
            "helpfulness": accuracy_result.get("helpfulness", 0.0),
            "retrieval_correctness": accuracy_result.get("retrieval_correctness", 0.0),
            "hit@k": accuracy_result.get("hit@k", 0.0),
            "retrieval_gate_pass_rate": retrieval_gate_pass_rate,
            "faithfulness_gate_pass_rate": faithfulness_gate_pass_rate,
            "hard_gate_pass_rate": hard_gate_pass_rate,
            "conditional_task_success": conditional_task_success_val,
            "instruction_following": instr_following_score,
            # "hit@k_failure_rate": accuracy_result.get("hit@k_failure_rate", 0.0),
            # "faithfulness_failure_rate": accuracy_result.get(
            #     "faithfulness_failure_rate", 0.0
            # ),
            "questions_evaluated": accuracy_result.get("questions_evaluated", 0)
            + if_result.get("questions_evaluated", 0),
            "results_dir": results_dir,
        }

        test_results = {
            "success": combined_result.get("success"),
            "results_dir": combined_result.get("results_dir"),
            "answer_correctness": combined_result.get("answer_correctness"),
            "faithfulness": combined_result.get("faithfulness"),
            "helpfulness": combined_result.get("helpfulness"),
            "retrieval_correctness": combined_result.get("retrieval_correctness"),
            "hit@k": combined_result.get("hit@k"),
            "conditional_task_success": combined_result.get("conditional_task_success"),
            "retrieval_gate_pass_rate": combined_result.get("retrieval_gate_pass_rate"),
            "faithfulness_gate_pass_rate": combined_result.get("faithfulness_gate_pass_rate"),
            "hard_gate_pass_rate": combined_result.get("hard_gate_pass_rate"),
            "conditional_task_success": combined_result.get("conditional_task_success"),
            "instruction_following": combined_result.get("instruction_following"),
        }

        if not combined_result.get("success"):
            test_results["error"] = accuracy_result.get("error") or if_result.get("error")

        output = {
            "test_results": test_results,
            "generated_reports": [write_html_report(detailed_results, test_results, results_dir)],
        }

        with open(os.path.join(results_dir, "output.json"), "w") as f:
            json.dump(output, f, indent=2)

        print(json.dumps(output, indent=2))

        # Exit with appropriate code
        sys.exit(0 if output["test_results"]["success"] else 1)

    except Exception as e:
        error_result = {
            "success": False,
            "results_dir": test_params.get("results_dir", "accuracy_rag_results")
            if "test_params" in locals()
            else "",
            "conditional_task_success": 0.0,
            "answer_correctness": 0.0,
            "retrieval_correctness": 0.0,
            "faithfulness": 0.0,
            "instruction_following": 0.0,
            "helpfulness": 0.0,
            "error": str(e),
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def write_html_report(detailed_results, test_results, results_dir):
    reports_dir = Path(results_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = "evaluation_report_by_metric.html"
    report_path = reports_dir / report_filename

    # Also generate metric-grouped JSON and a per-metric HTML report in the same results folder
    try:
        gen_mod = importlib.import_module("generate_records_by_metric")
        # Generate metric groups from in-memory detailed_results dict (no need to wait for file IO)
        metric_data = gen_mod.generate_groups_from_dict(
            detailed_results, include_metric_score=False
        )
        # Fire-and-forget: persist the JSON for consumers who want files, but don't block on it
        try:
            detailed_by_metric_path = (
                Path(results_dir) / "detailed_results_by_metric.json"
            )
            with open(detailed_by_metric_path, "w") as f:
                json.dump(metric_data, f, indent=2)
            print(f"Wrote metric-grouped results to {detailed_by_metric_path}")
        except Exception as e:
            print(
                f"Warning: failed to persist detailed_results_by_metric.json: {e}",
                file=sys.stderr,
            )
    except Exception as e:
        metric_data = None
        print(
            f"Warning: failed to generate detailed_results_by_metric in-memory: {e}",
            file=sys.stderr,
        )

    try:
        rep_mod = importlib.import_module("report_generator_by_metric")

        if metric_data is not None:
            try:
                report = rep_mod.create_report_by_metric(metric_data, test_results)
                import arakawa as ar

                # report_path = Path(results_dir) / "evaluation_report_by_metric.html"
                ar.save_report(report, path=str(report_path), standalone=True)
                print(f"Saved per-metric report to {report_path}")
            except Exception as e:
                print(
                    f"Error in create_report_by_metric or ar.save_report: {e}",
                    file=sys.stderr,
                )
                import traceback

                traceback.print_exc(file=sys.stderr)
        else:
            print(
                "Warning: metric_data is None, skipping HTML report generation",
                file=sys.stderr,
            )
    except Exception as e:
        print(
            f"Warning: failed to generate per-metric HTML report: {e}",
            file=sys.stderr,
        )

    return {
        "report_name": "detailed_report",
        "report_type": "html",
        "report_path": str(report_path),
    }


def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()