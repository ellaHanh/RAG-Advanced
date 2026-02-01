# Config Folder

The `config/` folder holds **static configuration** used by the application, notably versioned pricing for cost tracking.

---

## Contents

| File | Purpose |
|------|--------|
| **__init__.py** | Package marker; docstring describes the folder as containing versioned pricing and other static config. |
| **pricing.json** | Versioned pricing history for embedding and chat models. Each entry has an `effective_date` and a `models` map: per-model `input_per_1k`, `output_per_1k` (USD), and optional `description`. Used by `orchestration/pricing.py` (e.g. `PricingProvider`) to compute per-request cost for embeddings and LLM calls. |

---

## pricing.json structure

- **pricing_history**: List of pricing snapshots.
- Each snapshot has:
  - **effective_date**: ISO date/time from which this pricing applies.
  - **currency**: e.g. `"USD"`.
  - **models**: Object keyed by model name (e.g. `gpt-4o-mini`, `text-embedding-3-small`). Each model has:
    - **input_per_1k**: Cost per 1k input tokens.
    - **output_per_1k**: Cost per 1k output tokens (often 0 for embedding models).
    - **description**: Optional human-readable description.

The orchestration layer loads this (or equivalent) to provide a `PricingProvider` for the cost tracker so strategy execution can report `cost_usd` in API responses.

---

## Usage

Configuration is typically loaded at app startup (e.g. in `api/main.py` or orchestration initialization). You do not run scripts inside `config/` directly; the application reads `config/pricing.json` (or the path configured in the app) when building the pricing provider.

To change pricing: edit `pricing.json` and ensure the app uses the updated file (restart or reload config as designed by the project).
