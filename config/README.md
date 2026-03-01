# Config

Static configuration for the application, notably versioned pricing for cost tracking.

## Quickstart

No commands — the app loads config at startup. Edit `pricing.json` and restart (or reload) to apply changes.

## Features

- **pricing.json** — Versioned pricing history: `effective_date`, `currency`, `models` (per-model `input_per_1k`, `output_per_1k` in USD). Used by `orchestration/pricing.py` (`PricingProvider`) for per-request cost.

## Usage

Loaded at app startup (e.g. `api/main.py`, orchestration init). To change pricing: edit `config/pricing.json` and ensure the app uses the updated file (restart or reload as designed).

## Dependencies

None beyond project deps. Read by orchestration layer only.

## Related

- [Root README](../README.md)
- [orchestration/](../orchestration/README.md) — Cost tracker and pricing provider
