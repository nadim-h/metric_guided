# Replication Package: Metric-Guided Agentic Refactoring

Data and analysis for *From Cosmetic to Structural: A Paired Experiment on Metric-Guided Agentic Refactoring*.

An agent (`gptme` 0.31.0, Qwen3-Coder-30B-A3B-Instruct) refactored 24,620 competitive programming solutions (9,825 C++, 9,795 Java, 5,000 Python) twice:
- **Basic**: no tools.
- **Guided**: with the CodeScene MCP server 0.1.2.

That gives 49,240 runs.

## Contents

| Path | Contents |
| --- | --- |
| `data/` | Results archive and checksums. |
| `analysis/` | Scripts for all tables and figures. |
| `figures/` | Figures written by the scripts. |
| `prompt-templates/` | Basic (`prompt_basic.md`) and Guided (`prompt_mcp.md`) prompts. |
| `convs/` | The seven sampled agent conversations. |

## Run

From the repository root, with Python 3.14:

```bash
tar -xzf data/data.tar.gz && shasum -a 256 -c data/SHA256SUMS
pip install -r requirements.txt
```

Each script prints its results, labeled by the paper table they belong to.

| Script | Output |
| --- | --- |
| `dataset.py` | Dataset descriptives, SLoC by initial CH |
| `rq1_outcomes.py` | Pass, improved-CH, and effective refactoring rates; CH change |
| `rq2_operations.py` | Refactoring operations by level, smell counts, per-smell McNemar tests, `figures/smell_change.pdf` |
| `rq3_moderation.py` | GEE models, break-even points, file-size check, `figures/rq3_moderation.pdf` |
| `rq4_failures.py` | Failure composition, churn, error causes, `figures/failure_modes.pdf` |
| `rq5_cost.py` | Tokens and turns, cost per effective refactoring, routing simulation |
| `plot_movement.py` | Exploratory CH movement plot |

## Data

There is one JSON Lines file per language and condition. `mcp` in the name marks the Guided runs. Each line is one run:

| Field | Description |
| --- | --- |
| `name`, `task`, `ref-task` | Problem name, original code, refactored code |
| `task-review`, `ref-review` | CodeScene review: `score` (CH 1–10; 0 = not reviewable) and smells |
| `status` | Test outcome (`all-passed`, compile, runtime, test, or `exception` statuses) |
| `comp-err`, `rt-err` | Error messages |
| `agent-status` | `completed`, `no-diff`, `timeout`, or `process-failed` |
| `tokens-in`, `tokens-out`, `turns` | `gptme` session statistics |
| `miner` | RefactoringMiner 3.1.2 operations (Java and Python) |
