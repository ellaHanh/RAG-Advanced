"""
Persist benchmark report and detailed results to Postgres evaluation tables.

Writes to evaluation_runs, benchmark_results, and strategy_metrics so that
pipeline (or API) runs can be queried from the database. Requires the
evaluation schema extension to be applied (evaluation/schema_extension.sql).

Usage:
    from evaluation.db_persistence import persist_benchmark_to_db

    await persist_benchmark_to_db(
        pool, report, detailed_results, run_name="bioasq_run_1"
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from evaluation.benchmarks import BenchmarkReport

logger = logging.getLogger(__name__)


def _dict_to_jsonb(d: dict[int, float] | None) -> str:
    """Convert dict with int keys to JSON string for JSONB (keys as strings)."""
    if not d:
        return "{}"
    return json.dumps({str(k): v for k, v in d.items()})


async def persist_benchmark_to_db(
    pool: asyncpg.Pool,
    report: BenchmarkReport,
    detailed_results: list[dict[str, Any]],
    run_name: str | None = None,
    api_key_id: str | None = None,
) -> str:
    """
    Persist a benchmark report and per-query results to Postgres.

    Inserts one evaluation_run, one benchmark_result per strategy (aggregates),
    and one strategy_metric per (query, strategy) from detailed_results.

    Args:
        pool: asyncpg connection pool (same DB as main schema + evaluation extension).
        report: BenchmarkReport from BenchmarkRunner.run().
        detailed_results: List of per-query dicts with "query_id", "query",
            "ground_truth_chunk_ids", "results": { strategy_name: { ... } }.
        run_name: Optional label for the run (stored in config.run_name).
        api_key_id: Optional API key UUID for evaluation_runs.api_key_id.

    Returns:
        evaluation_run_id (UUID string).

    Raises:
        asyncpg.PostgresError: On DB errors.
    """
    config_json = report.config.model_dump() if report.config else {}
    if run_name:
        config_json["run_name"] = run_name
    results_json = report.to_dict()

    async with pool.acquire() as conn:
        run_id = await conn.fetchval(
            """
            INSERT INTO evaluation_runs (
                api_key_id, run_type, status, config, results,
                started_at, completed_at
            )
            VALUES ($1::uuid, 'benchmark', 'completed', $2::jsonb, $3::jsonb,
                    $4::timestamptz, $5::timestamptz)
            RETURNING id
            """,
            api_key_id,
            json.dumps(config_json),
            json.dumps(results_json),
            report.started_at,
            report.completed_at,
        )
        run_id_str = str(run_id)

        for strategy_name, stats in report.statistics.items():
            await conn.execute(
                """
                INSERT INTO benchmark_results (
                    evaluation_run_id, strategy_name,
                    avg_precision_at_k, avg_recall_at_k, avg_mrr, avg_ndcg_at_k,
                    latency_p50_ms, latency_p95_ms, latency_p99_ms,
                    total_cost_usd, avg_cost_per_query_usd,
                    query_count, iteration_count
                )
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6::jsonb,
                        $7, $8, $9, $10, $11, $12, $13)
                """,
                run_id,
                strategy_name,
                _dict_to_jsonb(stats.avg_precision),
                _dict_to_jsonb(stats.avg_recall),
                stats.avg_mrr,
                _dict_to_jsonb(stats.avg_ndcg),
                int(stats.latency_p50),
                int(stats.latency_p95),
                int(stats.latency_p99),
                stats.total_cost,
                stats.avg_cost_per_query,
                stats.query_count,
                stats.iteration_count,
            )

        for row in detailed_results:
            query_id = row.get("query_id") or ""
            results_per_strategy = row.get("results") or {}
            for strategy_name, sr in results_per_strategy.items():
                if not isinstance(sr, dict):
                    continue
                precision = sr.get("precision_at_k")
                recall = sr.get("recall_at_k")
                ndcg = sr.get("ndcg_at_k")
                await conn.execute(
                    """
                    INSERT INTO strategy_metrics (
                        evaluation_run_id, strategy_name, query_id,
                        precision_at_k, recall_at_k, mrr, ndcg_at_k,
                        latency_ms, cost_usd, retrieved_doc_ids
                    )
                    VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7::jsonb,
                            $8, $9, $10::text[])
                    """,
                    run_id,
                    strategy_name,
                    query_id,
                    _dict_to_jsonb(precision) if isinstance(precision, dict) else "{}",
                    _dict_to_jsonb(recall) if isinstance(recall, dict) else "{}",
                    float(sr.get("mrr") or 0),
                    _dict_to_jsonb(ndcg) if isinstance(ndcg, dict) else "{}",
                    int(sr.get("latency_ms") or 0),
                    float(sr.get("cost_usd") or 0),
                    sr.get("retrieved_chunk_ids") or [],
                )

    logger.info(
        "Persisted benchmark to DB: evaluation_run_id=%s, strategies=%d, rows=%d",
        run_id_str,
        len(report.statistics),
        sum(len(row.get("results") or {}) for row in detailed_results),
    )
    return run_id_str
