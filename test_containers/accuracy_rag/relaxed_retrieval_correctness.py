"""Simple utility: hit@k calculation.

This module provides a tiny, dependency-free implementation of hit@k used
for retrieval evaluation. The user requested a simple function to compute
hit@k with k=5; the default k is 5 but it can be overridden.
"""

from typing import List, Dict, Any


class RelaxedRetrievalCorrectness:
    """Hit@k evaluator for retrieval tasks."""

    @staticmethod
    def hit_at_k(
        retrieved_contexts: List[str], reference_context: List[str], k: int = 5
    ) -> Dict[str, Any]:
        """Compute hit@k for a single query.

        Args:
            retrieved_contexts: list of retrieved contexts (ordered by relevance, most-relevant first).
            reference_context: list of ground-truth reference contexts to match against.
            k: cutoff for top-k (default 5).

        Returns:
            Tuple[float, str]: (score, explanation)

                - score: 1.0 if any reference was found within the top-k retrieved contexts, otherwise 0.0
                - explanation: human-readable message describing the match (includes position when found)
        """
        topk = retrieved_contexts[:k]
        hit = any(ref in topk for ref in reference_context)

        if hit:
            matched_ref = next(ref for ref in reference_context if ref in topk)
            matched_pos = topk.index(matched_ref) + 1
            explanation = (
                f"Reference context found at position {matched_pos} within top-{k}"
            )
            # Generate simplified summary based on position
            if matched_pos <= 2:
                summary = "The most relevant context was retrieved."
            elif matched_pos <= k // 2:
                summary = "A relevant context was retrieved but ranked lower."
            else:
                summary = (
                    "A relevant context was retrieved but appears in the bottom half."
                )
        else:
            explanation = f"No reference context found in top-{k} retrieved contexts"
            summary = "No relevant context was retrieved."

        return {
            "score": 1.0 if hit else 0.0,
            "explanation_metadata": {"explanation": explanation},
            "explanation_summary": summary,
        }


# if __name__ == "__main__":
#     # Example: evaluate one query against reference contexts and retrieved contexts, one at a time
#     references = ["The capital of France is Paris"]

#     print("Test 1: one reference in position 1 (top-1)")
#     retrieved_1 = ["The capital of France is Paris", "Paris is in Europe", "France borders Germany", "Paris has the Eiffel Tower"]
#     result_1 = RetrievalCorrectness.hit_at_k(retrieved_1, references, k=5)
#     print(f"  References: {references}")
#     print(f"  Retrieved: {retrieved_1}")
#     print(f"  Result: {result_1}\n")

#     print("Test 2: one reference in position 3 (within top-5)")
#     retrieved_2 = ["Paris is in Europe", "France borders Germany", "The capital of France is Paris", "Paris has the Eiffel Tower"]
#     result_2 = RetrievalCorrectness.hit_at_k(retrieved_2, references, k=5)
#     print(f"  References: {references}")
#     print(f"  Retrieved: {retrieved_2}")
#     print(f"  Result: {result_2}\n")

#     print("Test 3: no references in top-5")
#     retrieved_3 = ["Paris is in Europe", "France borders Germany", "Paris has museums", "France is large"]
#     result_3 = RetrievalCorrectness.hit_at_k(retrieved_3, references, k=5)
#     print(f"  References: {references}")
#     print(f"  Retrieved: {retrieved_3}")
#     print(f"  Result: {result_3}\n")

#     print("Test 4: second reference in position 2 with k=3")
#     retrieved_4 = ["Paris is in Europe", "Paris is the capital", "France borders Germany"]
#     result_4 = RetrievalCorrectness.hit_at_k(retrieved_4, references, k=3)
#     print(f"  References: {references}")
#     print(f"  Retrieved: {retrieved_4}")
#     print(f"  Result: {result_4}")
