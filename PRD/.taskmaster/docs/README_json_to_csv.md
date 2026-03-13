# JSON-to-CSV Conversion

Convert Task Master JSON files to CSV for easier reading in spreadsheets or text editors.

**Script:** `.taskmaster/scripts/json_to_csv.py`

## Behavior

- **Output:** Same base name as the input file, `.csv` extension, written in the **same directory** as the input.
- **Default input:** `.taskmaster/tasks/tasks.json` (resolved relative to the script).
- **Explicit inputs:** One or more paths (absolute or relative). Each path is converted to a CSV in that file’s directory.

## Usage

```bash
# From repo root or any directory
python .taskmaster/scripts/json_to_csv.py

# Explicit path(s)
python .taskmaster/scripts/json_to_csv.py /path/to/tasks.json
python .taskmaster/scripts/json_to_csv.py \
  .taskmaster/tasks/tasks.json \
  .taskmaster/reports/task-complexity-report.json
```

With no arguments, the script converts the default `tasks.json` and prints the path of the written CSV. With one or more paths, it converts each and prints each output path.

## Supported Inputs

### 1. Tasks JSON (`tasks.json`)

Structure: `{ "master": { "tasks": [ ... ] } }`.

**Output CSV columns:**

| Column              | Description                                      |
|---------------------|--------------------------------------------------|
| `kind`              | `task` or `subtask`                              |
| `id`                | Task or subtask id                               |
| `parent_id`         | Parent task id (empty for tasks)                  |
| `title`             | Title                                            |
| `description`       | Description                                      |
| `details`           | Details                                          |
| `testStrategy`      | Test strategy                                    |
| `priority`          | Priority (tasks only)                            |
| `dependencies`      | Comma-separated dependency ids                   |
| `status`            | Status                                           |
| `complexity`         | Complexity score (tasks only)                    |
| `recommendedSubtasks` | Recommended subtasks count (tasks only)        |
| `expansionPrompt`   | Expansion prompt (tasks only)                     |
| `updatedAt`         | Last updated (tasks only)                        |

Tasks and subtasks are flattened into one table; subtasks have `kind=subtask` and `parent_id` set.

### 2. Task-Complexity-Report JSON (`task-complexity-report.json`)

Structure: `{ "meta": { ... }, "complexityAnalysis": [ ... ] }`.

**Output CSV columns:**

| Column               | Description                    |
|----------------------|--------------------------------|
| `taskId`             | Task id                        |
| `taskTitle`          | Task title                     |
| `complexityScore`    | Complexity score               |
| `recommendedSubtasks`| Recommended subtasks count     |
| `expansionPrompt`    | Expansion prompt               |
| `reasoning`          | Complexity reasoning           |
| `reportGeneratedAt`  | From `meta.generatedAt`        |
| `thresholdScore`     | From `meta.thresholdScore`     |
| `tasksAnalyzed`      | From `meta.tasksAnalyzed`      |

One row per entry in `complexityAnalysis`; report meta is repeated on each row.

## Requirements

- Python 3.8+
- No extra dependencies (uses only `json`, `csv`, `pathlib`, `argparse`).

## Example Output Paths

- Input: `.taskmaster/tasks/tasks.json` → Output: `.taskmaster/tasks/tasks.csv`
- Input: `.taskmaster/reports/task-complexity-report.json` → Output: `.taskmaster/reports/task-complexity-report.csv`
