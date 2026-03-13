"""
Run the evaluation pipeline: load gold + corpus from xlsx, ingest corpus, run benchmarks.

Uses the same strategy executor as the API. Supports config file (YAML/JSON) and
CLI overrides for column mapping. Outputs metrics summary, BenchmarkReport, and
optional JSON dataset and markdown/HTML reports.

Usage:
  python scripts/run_evaluation_pipeline.py \\
    --gold datasets/evaluation_gold_doc_bioasq_sampled_100_title_v1_2026Feb13.xlsx \\
    --corpus datasets/evaluation_corpus_bioasq_sampled_100_title_v1_2026Feb13.xlsx \\
    --strategies standard reranking \\
    --out-dir ./eval_out

  With custom column mapping (CLI):
  python scripts/run_evaluation_pipeline.py --gold gold.xlsx --corpus corpus.xlsx \\
    --gold-map query_id=id,query=question,relevant_doc_ids=relevant_passage_ids \\
    --corpus-map doc_id=doc_id,text=passage,title=title

  With config file:
  python scripts/run_evaluation_pipeline.py --config evaluation/config/bioasq_v1.yaml \\
    --gold gold.xlsx --corpus corpus.xlsx --out-dir ./eval_out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path

import asyncpg

# Add project root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkQuery,
    BenchmarkRunner,
    StrategyResult,
)
from evaluation.corpus_ingest import (
    doc_id_to_stem,
    get_doc_id_to_chunk_ids,
    ingest_corpus_and_get_chunk_map,
)
from evaluation.loaders.xlsx_config import (
    BIOASQ_V1_CORPUS_CONFIG,
    BIOASQ_V1_GOLD_CONFIG,
    XlsxCorpusConfig,
    XlsxGoldConfig,
)
from evaluation.loaders.xlsx_loader import load_corpus_xlsx, load_gold_dataset_from_xlsx
from evaluation.db_persistence import persist_benchmark_to_db
from evaluation.ragas_eval import RagasEvaluationResult, evaluate_generation
from evaluation.reports import save_report
from generation.chain import generate_answer
from orchestration.executor import StrategyExecutor
from orchestration.models import Document, StrategyConfig
from strategies.ingestion.models import IngestionConfig

logger = logging.getLogger(__name__)


def _make_json_serializable(obj: object) -> object:
    """Recursively convert numpy scalars to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
    except ImportError:
        pass
    return obj


def _normalize_strategy_name(name: str) -> str:
    """Convert CLI-friendly name to registry key (e.g. multi-query -> multi_query)."""
    return str(name).strip().replace("-", "_")


def _parse_map_arg(s: str) -> dict[str, str]:
    """Parse --gold-map or --corpus-map key=val,key2=val2 into dict."""
    out = {}
    for part in s.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# Pipeline default ingestion (same as corpus_ingest default when config=None)
PIPELINE_DEFAULT_INGESTION = IngestionConfig(
    chunk_size=2000,
    chunk_overlap=0,
    use_semantic_chunking=False,
    max_tokens=512,
)


def _load_pipeline_config(
    path: Path,
) -> tuple[XlsxGoldConfig | None, XlsxCorpusConfig | None, dict | None]:
    """Load gold, corpus, and optional ingestion config from YAML or JSON file.

    Returns:
        (gold_cfg, corpus_cfg, ingestion_overrides). ingestion_overrides is a dict
        with optional keys chunk_size, chunk_overlap, use_semantic_chunking, max_tokens,
        or None if no "ingestion" section is present.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            raise RuntimeError("PyYAML required for YAML config: pip install pyyaml")
    else:
        data = json.loads(text)
    gold_cfg = None
    corpus_cfg = None
    ingestion_overrides: dict | None = None
    if data:
        g = data.get("gold")
        if g:
            gold_cfg = XlsxGoldConfig(
                sheet_index=g.get("sheet_index", 0),
                query_id_column=g["query_id_column"],
                query_column=g["query_column"],
                relevant_doc_ids_column=g["relevant_doc_ids_column"],
                list_format=g.get("list_format", "json"),
                answer_column=g.get("answer_column"),
                gold_decision_column=g.get("gold_decision_column"),
                gold_explanation_column=g.get("gold_explanation_column"),
            )
        c = data.get("corpus")
        if c:
            corpus_cfg = XlsxCorpusConfig(
                sheet_index=c.get("sheet_index", 0),
                doc_id_column=c["doc_id_column"],
                text_column=c["text_column"],
                title_column=c.get("title_column"),
            )
        ing = data.get("ingestion")
        if isinstance(ing, dict) and ing:
            ingestion_overrides = {
                k: v for k, v in ing.items()
                if k in ("chunk_size", "chunk_overlap", "use_semantic_chunking", "max_tokens")
            }
            if not ingestion_overrides:
                ingestion_overrides = None
    return gold_cfg, corpus_cfg, ingestion_overrides


def _make_benchmark_executor(
    strategy_config: StrategyConfig | None = None,
    include_retrieved_contexts: bool = False,
):
    """Build an async executor (strategy_name, query_data) -> StrategyResult for BenchmarkRunner.

    When include_retrieved_contexts is True (e.g. for generation eval), StrategyResult
    will include retrieved_contexts (list of chunk content strings).
    """
    exec_config = strategy_config or StrategyConfig(limit=10)

    async def executor(strategy_name: str, query_data: dict) -> StrategyResult:
        from orchestration.executor import StrategyExecutor
        ex = StrategyExecutor()
        try:
            result = await ex.execute(
                strategy_name=strategy_name,
                query=query_data["query"],
                config=exec_config,
            )
            retrieved_contexts = None
            if include_retrieved_contexts and result.documents:
                retrieved_contexts = [d.content for d in result.documents]
            return StrategyResult(
                strategy_name=strategy_name,
                query_id=query_data.get("query_id", ""),
                retrieved_chunk_ids=result.document_ids,
                retrieved_contexts=retrieved_contexts,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                success=True,
            )
        except Exception as e:
            logger.exception("Strategy %s failed for query %s", strategy_name, query_data.get("query_id"))
            return StrategyResult(
                strategy_name=strategy_name,
                query_id=query_data.get("query_id", ""),
                retrieved_chunk_ids=[],
                retrieved_contexts=None,
                latency_ms=0,
                cost_usd=0.0,
                success=False,
                error=str(e),
            )
    return executor


async def _run_generation_eval(
    benchmark_queries: list[BenchmarkQuery],
    detailed_results: list[dict],
    strategies: list[str],
    generation_model: str | None,
) -> RagasEvaluationResult | None:
    """Run generation for each query (using first strategy's contexts), then RAGAS eval."""
    if not strategies or not detailed_results or len(detailed_results) != len(benchmark_queries):
        return None
    strategy = strategies[0]
    ragas_samples: list[dict] = []
    loop = asyncio.get_event_loop()
    for i, query in enumerate(benchmark_queries):
        gt = query.ground_truth_answer
        if not gt or not gt.strip():
            continue
        row = detailed_results[i]
        res = row.get("results", {}).get(strategy)
        if not res or not res.get("success"):
            continue
        contexts = res.get("retrieved_contexts")
        if not contexts or not isinstance(contexts, list):
            continue
        docs = [
            Document(id=str(j), content=ctx, title="", source="")
            for j, ctx in enumerate(contexts)
        ]
        try:
            gen_result = await loop.run_in_executor(
                None,
                lambda q=query.query, d=docs, m=generation_model: generate_answer(
                    q, d, model=m or None
                ),
            )
        except Exception as e:
            logger.warning("Generation failed for query %s: %s", query.query_id, e)
            continue
        ragas_samples.append({
            "question": query.query,
            "contexts": contexts,
            "answer": gen_result.answer,
            "ground_truth": gt,
        })
    if not ragas_samples:
        logger.warning("No RAGAS samples collected for generation eval")
        return RagasEvaluationResult(scores={})
    try:
        return await loop.run_in_executor(
            None,
            lambda: evaluate_generation(ragas_samples, show_progress=True),
        )
    except Exception as e:
        logger.exception("RAGAS evaluation failed: %s", e)
        return RagasEvaluationResult(scores={})


async def run_pipeline(
    gold_path: Path,
    corpus_path: Path,
    gold_config: XlsxGoldConfig | None = None,
    corpus_config: XlsxCorpusConfig | None = None,
    strategies: list[str] | None = None,
    out_dir: Path | None = None,
    limit: int = 10,
    iterations: int = 1,
    limit_queries: int | None = None,
    limit_corpus: int | None = None,
    extra_corpus: int | None = None,
    random_queries: bool = False,
    random_corpus: bool = False,
    seed: int = 42,
    skip_ingest: bool = False,
    clean_before_ingest: bool = True,
    save_json_dataset: bool = True,
    save_markdown_report: bool = True,
    save_detailed_results: bool = True,
    write_db: bool = False,
    run_name: str | None = None,
    ingestion_config: IngestionConfig | None = None,
    run_generation_eval: bool = False,
    generation_model: str | None = None,
) -> None:
    """
    Load gold and corpus, ingest corpus, run benchmarks, write outputs.

    Gold relevant_doc_ids (corpus doc_ids) are mapped to chunk IDs after ingest
    so metrics use retrieval chunk IDs.
    """
    strategies = [_normalize_strategy_name(s) for s in (strategies or ["standard"])]
    gold_config = gold_config or BIOASQ_V1_GOLD_CONFIG
    corpus_config = corpus_config or BIOASQ_V1_CORPUS_CONFIG

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL must be set for the evaluation pipeline")
    pool = await asyncpg.create_pool(
        database_url,
        min_size=1,
        max_size=5,
        command_timeout=120,
    )
    try:
        from strategies.agents import register_all_strategies
        from strategies.utils.embedder import embed_query
        register_all_strategies(pool, embed_query)
        logger.info("Registered strategies for benchmark")
    except Exception:
        await pool.close()
        raise

    if run_generation_eval:
        try:
            from orchestration.pricing import get_pricing_provider
            await get_pricing_provider()
            logger.info("Initialized pricing provider for RAGAS cost")
        except Exception as e:
            logger.warning("Could not initialize pricing provider for RAGAS cost: %s", e)

    try:
        await _run_pipeline_impl(
            pool,
            gold_path,
            corpus_path,
            gold_config,
            corpus_config,
            strategies,
            out_dir,
            limit,
            iterations,
            limit_queries,
            limit_corpus,
            extra_corpus,
            random_queries,
            random_corpus,
            seed,
            skip_ingest,
            clean_before_ingest,
            save_json_dataset,
            save_markdown_report,
            save_detailed_results,
            write_db=write_db,
            run_name=run_name,
            ingestion_config=ingestion_config,
            run_generation_eval=run_generation_eval,
            generation_model=generation_model,
        )
    finally:
        await pool.close()


async def _run_pipeline_impl(
    pool: asyncpg.Pool,
    gold_path: Path,
    corpus_path: Path,
    gold_config: XlsxGoldConfig | None,
    corpus_config: XlsxCorpusConfig | None,
    strategies: list[str],
    out_dir: Path | None,
    limit: int,
    iterations: int,
    limit_queries: int | None,
    limit_corpus: int | None,
    extra_corpus: int | None,
    random_queries: bool,
    random_corpus: bool,
    seed: int,
    skip_ingest: bool,
    clean_before_ingest: bool,
    save_json_dataset: bool,
    save_markdown_report: bool,
    save_detailed_results: bool,
    write_db: bool = False,
    run_name: str | None = None,
    ingestion_config: IngestionConfig | None = None,
    run_generation_eval: bool = False,
    generation_model: str | None = None,
) -> None:
    """Inner implementation; pool already created and strategies registered."""
    logger.info("Loading gold from %s", gold_path)
    dataset = load_gold_dataset_from_xlsx(gold_path, config=gold_config)
    logger.info("Gold dataset: %d queries", len(dataset.queries))

    rng = random.Random(seed) if (random_queries or random_corpus) else None
    if limit_queries is not None and random_queries and rng is not None:
        queries_to_use = rng.sample(
            dataset.queries, min(limit_queries, len(dataset.queries))
        )
        logger.info("Randomly sampled %d queries (seed=%d)", len(queries_to_use), seed)
    elif limit_queries is not None:
        queries_to_use = dataset.queries[:limit_queries]
        logger.info("Limited to first %d queries for quick run", len(queries_to_use))
    else:
        queries_to_use = dataset.queries

    if skip_ingest:
        stem_to_chunk_ids = await get_doc_id_to_chunk_ids(pool, source_suffix=".txt")
        doc_id_to_chunk_ids = {
            stem: chunk_ids for stem, chunk_ids in stem_to_chunk_ids.items()
        }
        logger.info(
            "Skipping ingestion: using existing DB map for %d doc stems",
            len(doc_id_to_chunk_ids),
        )
    else:
        logger.info("Loading corpus from %s", corpus_path)
        corpus = load_corpus_xlsx(corpus_path, config=corpus_config)
        logger.info("Corpus: %d documents", len(corpus))
        if extra_corpus is not None:
            gold_doc_ids = {
                doc_id for q in queries_to_use for doc_id in q.relevant_doc_ids
            }
            corpus_ids = {r["id"] for r in corpus}
            missing = gold_doc_ids - corpus_ids
            if missing:
                logger.warning(
                    "Gold references %d doc_id(s) not in corpus; recall may be undercounted: %s",
                    len(missing),
                    list(missing)[:5],
                )
            gold_rows = [r for r in corpus if r["id"] in gold_doc_ids]
            non_gold = [r for r in corpus if r["id"] not in gold_doc_ids]
            if random_corpus and rng is not None:
                extra_rows = rng.sample(
                    non_gold, min(extra_corpus, len(non_gold))
                )
                logger.info(
                    "Corpus: %d gold docs + %d extra (random, seed=%d)",
                    len(gold_rows),
                    len(extra_rows),
                    seed,
                )
            else:
                extra_rows = non_gold[:extra_corpus]
                logger.info(
                    "Corpus: %d gold docs + %d extra (first non-gold)",
                    len(gold_rows),
                    len(extra_rows),
                )
            corpus = gold_rows + extra_rows
        elif limit_corpus is not None:
            corpus = corpus[:limit_corpus]
            logger.info("Limited corpus to %d documents for quick run", len(corpus))

        logger.info("Ingesting corpus (clean_before=%s)...", clean_before_ingest)

        def progress(i: int, total: int) -> None:
            if total and i % max(1, total // 10) == 0:
                logger.info("Ingestion progress %d/%d", i, total)

        doc_id_to_chunk_ids_raw = await ingest_corpus_and_get_chunk_map(
            corpus,
            config=ingestion_config,
            clean_before=clean_before_ingest,
            progress_cb=progress,
        )
        doc_id_to_chunk_ids = doc_id_to_chunk_ids_raw
        logger.info(
            "Built doc_id -> chunk_id map for %d corpus docs",
            len(doc_id_to_chunk_ids),
        )

    # Map gold doc IDs to chunk IDs and build benchmark queries (chunk-level evaluation)
    benchmark_queries: list[BenchmarkQuery] = []
    for q in queries_to_use:
        chunk_ids: list[str] = []
        for doc_id in q.relevant_doc_ids:
            key = doc_id_to_stem(doc_id) if skip_ingest else doc_id
            chunk_ids.extend(doc_id_to_chunk_ids.get(key, []))
        benchmark_queries.append(
            BenchmarkQuery(
                query_id=q.query_id,
                query=q.query,
                ground_truth_chunk_ids=chunk_ids,
                relevance_scores=q.relevance_scores,
                ground_truth_answer=q.answer,
                gold_decision=q.gold_decision,
                gold_explanation=q.gold_explanation,
            )
        )

    if not benchmark_queries:
        logger.warning("No benchmark queries after mapping; check gold relevant_passage_ids vs corpus doc_id")
        return

    if run_generation_eval:
        if gold_config and gold_config.answer_column is None:
            logger.warning(
                "run_generation_eval is True but gold config has no answer_column; "
                "generation eval will be skipped. Set answer_column in config (e.g. bioasq_v1.json)."
            )
            run_generation_eval = False
        else:
            with_gt = sum(1 for q in benchmark_queries if q.ground_truth_answer and q.ground_truth_answer.strip())
            if with_gt == 0:
                logger.warning(
                    "run_generation_eval is True but no query has ground_truth_answer; "
                    "generation eval will be skipped."
                )
                run_generation_eval = False
            else:
                logger.info("Generation eval enabled: %d queries have ground truth answer", with_gt)

    exec_config = StrategyConfig(limit=limit)
    runner = BenchmarkRunner(
        executor=_make_benchmark_executor(
            exec_config,
            include_retrieved_contexts=run_generation_eval,
        ),
    )
    config = BenchmarkConfig(
        strategies=strategies,
        iterations=iterations,
        k_values=[3, 5, 10],
        timeout_seconds=60.0,
        max_concurrent=2,
    )

    logger.info("Running benchmark: strategies=%s, queries=%d", strategies, len(benchmark_queries))
    report, detailed_results = await runner.run(benchmark_queries, config)
    logger.info("Benchmark completed in %.2fs", report.duration_seconds)
    print(report.summary())

    generation_metrics_result = None
    if run_generation_eval and detailed_results:
        generation_metrics_result = await _run_generation_eval(
            benchmark_queries=benchmark_queries,
            detailed_results=detailed_results,
            strategies=strategies,
            generation_model=generation_model,
        )
        if generation_metrics_result and generation_metrics_result.scores:
            print("\nRAGAS generation metrics:", generation_metrics_result.scores)
            if generation_metrics_result.llm_usage or generation_metrics_result.embedding_usage:
                parts = []
                if generation_metrics_result.llm_usage:
                    tu = generation_metrics_result.llm_usage
                    parts.append(
                        f"LLM: {tu.get('total_tokens', 0)} tokens"
                        + (f", ${tu.get('cost_usd')}" if tu.get('cost_usd') is not None else "")
                    )
                if generation_metrics_result.embedding_usage:
                    eu = generation_metrics_result.embedding_usage
                    parts.append(f"embedding: {eu.get('request_count', 0)} requests, {eu.get('texts_embedded', 0)} texts")
                if parts:
                    print("RAGAS usage:", " | ".join(parts))

    if write_db:
        try:
            eval_run_id = await persist_benchmark_to_db(
                pool, report, detailed_results or [], run_name=run_name
            )
            logger.info("Wrote benchmark to DB: evaluation_run_id=%s", eval_run_id)
        except Exception as e:
            logger.exception("Failed to persist benchmark to DB: %s", e)

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_detailed_results and detailed_results:
            detail_path = out_dir / "benchmark_results_detailed.json"
            detail_path.write_text(
                json.dumps(detailed_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved per-query per-strategy results to %s", detail_path)
        if save_json_dataset:
            json_path = out_dir / "gold_dataset_canonical.json"
            canonical = {
                "name": dataset.name,
                "description": dataset.description,
                "queries": [q.to_dict() for q in dataset.queries],
                "metadata": dataset.metadata,
            }
            json_path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved canonical dataset to %s", json_path)
        if save_markdown_report:
            md_path = out_dir / "benchmark_report.md"
            await save_report(report, str(md_path))
            logger.info("Saved markdown report to %s", md_path)
        if generation_metrics_result is not None and generation_metrics_result.scores:
            gen_path = out_dir / "generation_metrics.json"
            payload = {
                "scores": generation_metrics_result.scores,
                "per_sample": generation_metrics_result.per_sample,
            }
            if generation_metrics_result.llm_usage is not None:
                payload["ragas_llm_usage"] = generation_metrics_result.llm_usage
            if generation_metrics_result.embedding_usage is not None:
                payload["ragas_embedding_usage"] = generation_metrics_result.embedding_usage
            gen_path.write_text(
                json.dumps(_make_json_serializable(payload), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved RAGAS generation metrics to %s", gen_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline: load gold + corpus xlsx, ingest, benchmark, report.",
    )
    parser.add_argument("--gold", required=True, help="Path to gold xlsx (query + relevant_passage_ids)")
    parser.add_argument("--corpus", required=True, help="Path to corpus xlsx (doc_id, passage, title)")
    parser.add_argument("--config", type=Path, help="Optional YAML/JSON config for column mapping")
    parser.add_argument(
        "--gold-map",
        help="Gold column map: query_id=id,query=question,relevant_doc_ids=relevant_passage_ids",
    )
    parser.add_argument(
        "--corpus-map",
        help="Corpus column map: doc_id=doc_id,text=passage,title=title",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["standard"],
        help="Strategies to benchmark (use underscores or hyphens, e.g. multi_query or multi-query)",
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory for report and JSON dataset")
    parser.add_argument("--limit", type=int, default=10, help="Retrieval limit per strategy")
    parser.add_argument("--iterations", type=int, default=1, help="Benchmark iterations per query")
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        metavar="N",
        help="Use only first N queries (quick pipeline test)",
    )
    parser.add_argument(
        "--limit-corpus",
        type=int,
        default=None,
        metavar="N",
        help="Ingest only first N corpus documents (quick pipeline test)",
    )
    parser.add_argument(
        "--extra-corpus",
        type=int,
        default=None,
        metavar="X",
        help="Include all corpus docs referenced in gold (for limited queries) plus X extra docs. Ensures metrics are reliable; --limit-corpus is ignored when set.",
    )
    parser.add_argument(
        "--random-queries",
        action="store_true",
        help="With --limit-queries N, randomly sample N queries instead of first N (use --seed for reproducibility).",
    )
    parser.add_argument(
        "--random-corpus",
        action="store_true",
        help="With --extra-corpus X, randomly sample X extra docs from non-gold corpus instead of first X (use --seed for reproducibility).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="RNG seed for --random-queries and/or --random-corpus (default: 42).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse existing DB: do not ingest corpus; build doc_id->chunk_ids from current DB. Use after a full run with same corpus to re-run benchmarks only.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not clean DB before corpus ingest")
    parser.add_argument("--no-json", action="store_true", help="Do not save canonical JSON dataset")
    parser.add_argument("--no-report", action="store_true", help="Do not save markdown report")
    parser.add_argument(
        "--no-detailed-results",
        action="store_true",
        help="Do not save per-query per-strategy detailed JSON (benchmark_results_detailed.json)",
    )
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Persist benchmark to Postgres (evaluation_runs, benchmark_results, strategy_metrics). Requires evaluation schema extension.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Optional label for the run when using --write-db (stored in evaluation_runs.config).",
    )
    parser.add_argument(
        "--run-generation-eval",
        action="store_true",
        help="Run RAG generation and RAGAS evaluation. Requires gold xlsx to have answer column (see config answer_column).",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        metavar="MODEL",
        help="LLM model for generation (default: GENERATION_MODEL env or gpt-4o-mini).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Ingestion chunk size in characters (overrides config file; default 2000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Ingestion chunk overlap in characters (overrides config file; default 0).",
    )
    parser.add_argument(
        "--semantic-chunking",
        action="store_true",
        help="Use Docling HybridChunker for chunking (including plain text via HTML conversion).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Max tokens per chunk for Docling HybridChunker (overrides config file; default 512).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    gold_path = Path(args.gold)
    corpus_path = Path(args.corpus)
    if not gold_path.exists():
        logger.error("Gold file not found: %s", gold_path)
        raise SystemExit(1)
    if not corpus_path.exists():
        logger.error("Corpus file not found: %s", corpus_path)
        raise SystemExit(1)

    gold_config: XlsxGoldConfig | None = None
    corpus_config: XlsxCorpusConfig | None = None
    ingestion_overrides: dict | None = None
    if args.config and args.config.exists():
        gold_config, corpus_config, ingestion_overrides = _load_pipeline_config(args.config)
    if args.gold_map:
        m = _parse_map_arg(args.gold_map)
        gold_config = XlsxGoldConfig(
            query_id_column=m.get("query_id", "id"),
            query_column=m.get("query", "question"),
            relevant_doc_ids_column=m.get("relevant_doc_ids", "relevant_passage_ids"),
        )
    if args.corpus_map:
        m = _parse_map_arg(args.corpus_map)
        corpus_config = XlsxCorpusConfig(
            doc_id_column=m.get("doc_id", "doc_id"),
            text_column=m.get("text", "passage"),
            title_column=m.get("title"),
        )

    # Build ingestion config: pipeline default, then config file, then CLI (CLI overrides file).
    ingestion_config: IngestionConfig | None = None
    if ingestion_overrides or args.chunk_size is not None or args.chunk_overlap is not None or args.semantic_chunking or args.max_tokens is not None:
        base = PIPELINE_DEFAULT_INGESTION.model_dump()
        if ingestion_overrides:
            base = {**base, **ingestion_overrides}
        if args.chunk_size is not None:
            base["chunk_size"] = args.chunk_size
        if args.chunk_overlap is not None:
            base["chunk_overlap"] = args.chunk_overlap
        if args.semantic_chunking:
            base["use_semantic_chunking"] = True
        if args.max_tokens is not None:
            base["max_tokens"] = args.max_tokens
        ingestion_config = IngestionConfig(**base)

    strategies = [_normalize_strategy_name(s) for s in args.strategies]
    asyncio.run(
        run_pipeline(
            gold_path=gold_path,
            corpus_path=corpus_path,
            gold_config=gold_config,
            corpus_config=corpus_config,
            strategies=strategies,
            out_dir=args.out_dir,
            limit=args.limit,
            iterations=args.iterations,
            limit_queries=args.limit_queries,
            limit_corpus=args.limit_corpus,
            extra_corpus=args.extra_corpus,
            random_queries=args.random_queries,
            random_corpus=args.random_corpus,
            seed=args.seed,
            skip_ingest=args.skip_ingest,
            clean_before_ingest=not args.no_clean,
            save_json_dataset=not args.no_json,
            save_markdown_report=not args.no_report,
            save_detailed_results=not args.no_detailed_results,
            write_db=args.write_db,
            run_name=args.run_name,
            ingestion_config=ingestion_config,
            run_generation_eval=args.run_generation_eval,
            generation_model=args.generation_model,
        )
    )


if __name__ == "__main__":
    main()
