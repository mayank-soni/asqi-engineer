# Accuracy RAG test container — quick start

The container evaluates Retrieval-Augmented Generation (RAG) systems across the following metrics:

- **Answer Correctness** - Factual accuracy of responses
- **Faithfulness** - Alignment with retrieved context
- **Retrieval Correctness** - Quality of retrieved vs reference context
- **Hit@k** - Binary indicator if reference found in top-k results
- **Helpfulness** - Relevance and completeness of response
- **Instruction Following** - Compliance with transformed instructions
- **Task Success** - Composite metric combining Answer Correctness and Helpfulness

## Build the container
From this directory (`test_containers/accuracy_rag/`) you can build the test container. Two common options:

- Quick build (uses Docker cache):

```bash
docker build -t asqiengineer/test-container:accuracy_rag-latest .
```

- Clean build (no cache — recommended after dependency changes):

```bash
docker build --no-cache -t asqiengineer/test-container:accuracy_rag-latest .
```

Use the clean build when you changed Python dependencies or the Dockerfile; use the cached build for fast iterative edits that don't touch dependencies.

## Input datasets
Place your datasets in a directory and mount that directory as `/input` when running the container (or when running via ASQI). 

Example datasets can be found here: https://drive.google.com/drive/folders/11cuoTrxGeHtgJiq1ZYiopvEi9lkriHe4?usp=drive_link


Expected input layout (example):

- `/input/accuracy_rag/base_accuracy_dataset.jsonl`
- `/input/accuracy_rag/if_questions_dataset.csv`

Each dataset format is specific to the test suite. Ensure your files match the expected schema used by the container.

## Run via ASQI
Use ASQI CLI to run the test container and produce ASQI-standard outputs.

- Run tests only (produce raw test results):

```bash
asqi execute-tests \
  -t config/suites/sia_accuracy_test_suite.yaml \
  -s config/systems/sia_systems.yaml \
  -o test_results.json
```

- Run the full pipeline (tests + score card grading):

```bash
asqi execute \
  -t config/suites/sia_accuracy_test_suite.yaml \
  -s config/systems/sia_systems.yaml \
  -r config/score_cards/accuracy_rag_score_card.yaml \
  -o complete_results.json
```

These commands mount and handle volumes according to your ASQI configuration. By default ASQI writes outputs under a `results_dir` inside the container.

## Output
The container writes results under the configured `results_dir` inside the container (typically `/output/<results_dir>` on the host). Key files:

- `output.json` — aggregated system-level metrics (summary). Example:

```json
{
  "success": true,
  "results_dir": "/output/sia_accuracy_results",
  "answer_correctness": 0.95,
  "faithfulness": 0.90,
  "helpfulness": 0.92,
  "retrieval_correctness": 0.88,
  "hit@k": 0.85,
  "conditional_task_success": 0.90,
  "retrieval_gate_pass_rate": 0.95,
  "faithfulness_gate_pass_rate": 0.93,
  "hard_gate_pass_rate": 0.90,
  "instruction_following": 0.88
}
```

- `detailed_results.json` — per-sample results grouped by dataset (fields include: question, sut_answer, reference_contexts, retrieved_contexts, metrics).
- `detailed_results_by_metric.json` — sample-level results grouped by metric (used to generate per-metric reports).
- `reports/evaluation_report_by_metric.html` — interactive HTML report.

## Licensing requirements - TBD

The code in instructions_registry.py, instructions_util.py and instructions.py is originally taken from https://github.com/google-research/google-research/tree/master/instruction_following_eval and is licensed under the Apache License, Version 2.0 (the "License"). We have made minor modifications to the code. We need to ensure that the Licensing requirements are met before deployment.