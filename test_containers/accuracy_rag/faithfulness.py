import typing as t
from ragas.prompt import PydanticPrompt
from pydantic import BaseModel, Field
from ragas.metrics._faithfulness import (
    NLIStatementInput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorInput,
    StatementGeneratorOutput,
)
from dataclasses import dataclass, field
from ragas.metrics import Faithfulness
from langchain_core.callbacks import Callbacks
from ragas.dataset_schema import SingleTurnSample
import numpy as np


async def evaluate_faithfulness(
    user_input, response, retrieved_contexts, evaluator_llm
):
    sample = SingleTurnSample(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
    )

    scorer = MyFaithfulnessWithInstructions(llm=evaluator_llm)

    result = await scorer.single_turn_ascore(
        sample,
    )
    return result


class MyNLIStatementInput(BaseModel):
    user_instruction: str = Field(
        ..., description="The user instruction for the NLI task"
    )
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")


class MyStatementGeneratorPrompt(
    PydanticPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    instruction = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Each statement should:
    1. Be a complete, factual claim that can be independently verified
    2. Preserve all relevant context and relationships from the original answer
    3. Avoid ambiguous references by including necessary contextual information
    4. Replace all pronouns with the specific entities they refer to
    5. Maintain the logical connections between related pieces of information
    6. Be precise about quantities, conditions, and scope
    7. Be understandable without needing to reference other statements or the original context."""

    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is widely acknowledged to be one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        ),
        (
            StatementGeneratorInput(
                question="What were the deaths reported in the study?",
                answer="Overall, 6 deaths due to AEs were reported in the study (1 participant in the pembrolizumab group and 5 participants in the placebo group). Of those, 0 deaths were considered to be drug-related by the investigator.",
            ),
            StatementGeneratorOutput(
                statements=[
                    "6 deaths due to adverse events (AEs) were reported in the study.",
                    "1 death due to adverse events occurred in a participant from the pembrolizumab group.",
                    "5 deaths due to adverse events occurred in participants from the placebo group.",
                    "0 deaths out of the 6 reported deaths were considered to be drug-related by the investigator.",
                ]
            ),
        ),
    ]


class MyNLIStatementPrompt(PydanticPrompt[NLIStatementInput, NLIStatementOutput]):
    instruction = """Your task is to judge the faithfulness of a series of statements based on a given context and user instruction. For each statement you must return verdict as 1 if the statement satisfies one of the following conditions, or 0 if it does not:
    1. The statement can be directly inferred from the context
    2. The statement represents a logical conclusion that follows from applying the user instruction to the context, even if not explicitly stated in the context
    3. The statement is a reasonable interpretation or analysis required by the user instruction, supported by the available context
    
    When evaluating faithfulness:
    - Consider whether the statement aligns with what the user instruction asks for
    - Check if the statement is a logical consequence of following the user instruction given the context
    - Assess whether the statement represents appropriate reasoning or conclusions drawn from the context in response to the user instruction
    - A statement can be faithful even if it's not explicitly in the context, as long as it's a reasonable outcome of applying the user instruction to the context
    - Focus on the core logical accuracy and reasoning rather than minor semantic variations (e.g., "spent" vs "pledged", "similar" vs "comparable")
    - If the numerical data and logical conclusions are correct according to the instruction, minor word choice differences should not affect the faithfulness verdict"""

    input_model = NLIStatementInput
    output_model = NLIStatementOutput

    examples = [
        (
            NLIStatementInput(
                context="""Company A Q3 Earnings: Revenue increased by 8% year-over-year to $2.5 billion. Net income was $150 million, representing a 7 percent increase.
                Company B Q3 Earnings: Revenue grew 9% year-over-year to $1.8 billion. Net income reached $120 million, up 6percent from last year.""",
                user_instruction="Summarize the earnings report. If the earnings growth is less than 10 percent, mention that the companies' performance is comparable.",
                statements=[
                    "Company A had revenue of $2.5 billion with 8 percent growth.",
                    "Both companies' performance is comparable.",
                    "Company B outperformed Company A significantly.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Company A had revenue of $2.5 billion with 8 percent growth.",
                        reason="This information is directly stated in the context about Company A's Q3 earnings.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Both companies' performance is comparable.",
                        reason="While not explicitly stated in the context, this conclusion follows logically from the user instruction. Both companies had growth rates below 10% (8 percent and 9 percent revenue growth), and the instruction specifies to mention comparable performance when growth is less than 10%.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Company B outperformed Company A significantly.",
                        reason="The context shows Company B had 9 percent revenue growth vs Company A's 8%, which is not a significant difference. This contradicts the user instruction's logic that sub-10 percent growth rates should be characterized as comparable.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="""Company A has an average of 5 accidents per month, while Company B has an average of 12 accidents per month. Both companies have implemented safety measures to reduce accidents. Company A has spent 10 million on safety measures, while Company B has spent 15 million.""",
                user_instruction="Compare the accident rates between company A and company B. If the absolute difference in accident rates is less than 10 per month, say they are similar.",
                statements=[
                    "Company A has 5 accidents per month and has pledged 10 million to reduce this.",
                    "Company B has 12 accidents per month and has pledged 15 million to tackle this.",
                    "The accident rates of both companies are similar.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Company A has 5 accidents per month and has pledged 10 million to reduce this.",
                        reason="The numerical data (5 accidents, 10 million) is accurate from the context. While the context says 'spent' rather than 'pledged', this is a minor semantic variation that doesn't affect the core accuracy of the information.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Company B has 12 accidents per month and has pledged 15 million to tackle this.",
                        reason="The numerical data (12 accidents, 15 million) is accurate from the context. The minor word choice difference ('pledged' vs 'spent') doesn't impact the core factual accuracy.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="The accident rates of both companies are similar.",
                        reason="This follows logically from the user instruction. The absolute difference is |5-12| = 7, which is less than 10, so according to the instruction they should be characterized as similar.",
                        verdict=1,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="""John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
                user_instruction="Provide a summary of John's academic situation.",
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly mentioned as Computer Science in the context. This statement contradicts the given information.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason="The context mentions the specific courses John is enrolled in (Data Structures, Algorithms, and Database Management), and Artificial Intelligence is not among them. The user instruction asks for a summary of his academic situation, which should be based on the provided information.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason="The context states that John spends significant time studying, completing assignments, and staying late in the library for projects. This directly supports the characterization of him as dedicated, which is appropriate for summarizing his academic situation as requested.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information in the context about John having a part-time job, and this is not relevant to summarizing his academic situation as requested in the user instruction.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="The quarterly sales report shows: Product X sold 1,000 units, Product Y sold 800 units, Product Z sold 1,200 units.",
                user_instruction="Analyze the sales data and identify the best-performing product.",
                statements=[
                    "Product Z is the best-performing product.",
                    "Product Y had the lowest sales.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Product Z is the best-performing product.",
                        reason="While the context doesn't explicitly state which product is 'best-performing,' the user instruction asks to identify the best-performing product. Product Z has the highest sales (1,200 units), making this a logical conclusion based on the instruction and context.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="Product Y had the lowest sales.",
                        reason="This can be directly inferred from the context, which shows Product Y sold 800 units compared to 1,000 and 1,200 for the other products. This supports the analysis requested in the user instruction.",
                        verdict=1,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInput(
                context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
                user_instruction="Explain the scientific concept mentioned.",
                statements=[
                    "Albert Einstein was a genius.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="Albert Einstein was a genius.",
                        reason="This statement is completely unrelated to both the context about photosynthesis and the user instruction to explain the scientific concept. It neither derives from the context nor addresses the user's request.",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


@dataclass
class MyFaithfulnessWithInstructions(Faithfulness):
    nli_statements_prompt_modified: PydanticPrompt = field(
        default_factory=MyNLIStatementPrompt
    )
    statement_generator_prompt_modified: PydanticPrompt = field(
        default_factory=MyStatementGeneratorPrompt
    )

    async def _create_statements_modified(
        self, row: t.Dict, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        text, question = row["response"], row["user_input"]

        prompt_input = StatementGeneratorInput(question=question, answer=text)
        statements = await self.statement_generator_prompt_modified.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )
        return statements

    async def _create_verdicts_modified(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"
        user_instruction = row["user_input"]
        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_prompt_modified.generate(
            data=MyNLIStatementInput(
                user_instruction=user_instruction,
                context=contexts_str,
                statements=statements,
            ),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _ascore(self, row, callbacks) -> dict:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements_modified(row, callbacks)
        statements = statements.statements
        if statements == []:
            return {
                "score": np.nan,
                "explanation_metadata": {"verdicts": []},
                "explanation_summary": "No statements could be extracted from the response.",
            }

        verdictsmod = await self._create_verdicts_modified(row, statements, callbacks)
        verdictsdict = []
        for v in verdictsmod.statements:
            verdictsdict.append(
                {"statement": v.statement, "reason": v.reason, "verdict": v.verdict}
            )

        score = self._compute_score(verdictsmod)
        faithful_count = sum(v["verdict"] for v in verdictsdict)
        total_statements = len(verdictsdict)

        # Generate simplified summary explanation based on score
        if score >= 0.7:
            summary = "The model's answer is grounded in the retrieved contexts."
        elif score >= 0.4:
            summary = (
                "Some of the model's answer lacks grounding in the retrieved contexts."
            )
        else:
            summary = (
                "Much of the model's answer is not grounded in the retrieved contexts."
            )
        summary += (
            f" {faithful_count} out of {total_statements} statements were faithful."
        )

        return {
            "score": score,
            "explanation_metadata": {"verdicts": verdictsdict},
            "explanation_summary": summary,
        }
