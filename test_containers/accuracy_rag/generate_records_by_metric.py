"""Generate a JSON file grouping individual records by metric name.

Produces `detailed_results_by_metric.json` where the top-level keys are
metric names and values are lists of records associated with that metric.

Generic Multi-Dataset Approach:
- Discovers all datasets in detailed_results.json (e.g., base_accuracy_dataset, instruction_following_dataset)
- All records have a "metrics" dict with score/explanation_summary/explanation_metadata structure
- Groups ALL records from ALL datasets by metric name, maintaining canonical order
- Works with any number of datasets - no configuration needed!

This helper is intentionally standalone and does not change the HTML
report generator.
"""

import json
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Canonical metric order (must match LAYER1_METRIC_DESCRIPTIONS in report_generator_by_metric.py)
CANONICAL_METRIC_ORDER = [
    "answer_correctness",
    "faithfulness",
    "helpfulness",
    "retrieval_correctness",
    "hit@k",
    "retrieval_hard_gate",
    "faithfulness_hard_gate",
    "hard_gate_pass",
    "task_success",
    "instruction_following",
]


def main(
    detailed_results_path: Path = HERE / "detailed_results.json",
    out_path: Path = HERE / "detailed_results_by_metric.json",
    include_metric_score: bool = False,
) -> None:
    if not detailed_results_path.exists():
        raise FileNotFoundError(f"Could not find {detailed_results_path}")

    with open(detailed_results_path, "r") as f:
        data = json.load(f)

    # Delegate grouping logic to the in-memory helper to avoid duplication
    metric_groups = generate_groups_from_dict(
        data, include_metric_score=include_metric_score
    )

    # Write output
    with open(out_path, "w") as f:
        json.dump(metric_groups, f, indent=2)

    print(f"Wrote metric-grouped results to {out_path}")


def generate_groups_from_dict(
    detailed_results: dict, include_metric_score: bool = False
) -> dict:
    """Generate and return metric-grouped dict from an in-memory detailed_results dict.

    Generic approach for any multi-dataset evaluation:
    - Collects ALL datasets from detailed_results, maintaining order from file
    - Each record has a "metrics" dict with metric_name -> {score, explanation_summary, explanation_metadata}
    - Groups all records across datasets by metric name in canonical order
    - Works with any number of datasets as long as they follow the metrics structure

    This mirrors the behavior of main() but operates on a dict and returns the
    metric_groups dict instead of writing to disk. Caller may choose to persist
    the returned dict themselves.
    """
    # Collect all datasets in order they appear in detailed_results.json
    all_records_by_dataset = []
    all_metric_names = set()
    
    for key, value in detailed_results.items():
        if isinstance(value, list) and value:
            # Check if this is a dataset with records that have metrics
            if isinstance(value[0], dict) and "metrics" in value[0]:
                all_records_by_dataset.append((key, value))
                # Collect metric names from all records in this dataset
                for rec in value:
                    metrics = rec.get("metrics") or {}
                    if isinstance(metrics, dict):
                        all_metric_names.update(metrics.keys())
    
    if not all_records_by_dataset:
        # Return empty structure if no valid datasets found
        return {}

    # Order metrics: use canonical order, then append any unknown metrics alphabetically
    ordered_metrics = []
    for metric in CANONICAL_METRIC_ORDER:
        if metric in all_metric_names:
            ordered_metrics.append(metric)
    
    # Add any metrics not in canonical order (alphabetically sorted)
    remaining_metrics = sorted(all_metric_names - set(CANONICAL_METRIC_ORDER))
    ordered_metrics.extend(remaining_metrics)

    # Build metric_groups by iterating through each metric
    metric_groups = {}
    for m in ordered_metrics:
        metric_groups[m] = []
        
        # Collect records from all datasets for this metric, maintaining dataset order
        for dataset_key, records in all_records_by_dataset:
            for rec in records:
                # Only include this record if it actually has this metric
                if "metrics" not in rec or not isinstance(rec["metrics"], dict):
                    continue
                
                if m not in rec["metrics"]:
                    continue
                
                r = deepcopy(rec)
                r_metrics = {}
                
                # Extract the specific metric from this record
                m_info = rec["metrics"].get(m)
                if isinstance(m_info, dict):
                    r_metrics[m] = deepcopy(m_info)
                
                r["metrics"] = r_metrics

                if include_metric_score:
                    score = None
                    if m in r_metrics:
                        score = r_metrics[m].get("score")
                    r["metric_score_for"] = {"metric": m, "score": score}
                metric_groups[m].append(r)

    return metric_groups


if __name__ == "__main__":
    main()
