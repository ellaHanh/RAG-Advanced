"""
RAG-Advanced Evaluation Module.

IR metrics calculation, benchmarking, and report generation.

Exports:
    Metrics:
        - calculate_metrics: Calculate Precision@k, Recall@k, MRR, NDCG@k
        - calculate_batch_metrics: Calculate metrics for multiple queries
        - EvaluationMetrics: Container for calculated metrics

    Benchmarks:
        - BenchmarkRunner: Run multi-query benchmarks with statistical analysis
        - BenchmarkConfig: Configuration for benchmark runs
        - BenchmarkQuery: Single query in benchmark dataset
        - BenchmarkReport: Complete benchmark report
        - StrategyResult: Result from strategy execution
        - StrategyStatistics: Statistical summary for a strategy

    Datasets:
        - DatasetManager: Load, validate, and manage test datasets
        - Dataset: Collection of queries with ground truth
        - DatasetQuery: Single query with ground truth
        - DatasetConfig: Configuration for dataset loading
        - DatasetSplit: Train/test split result
        - load_dataset: Convenience function to load dataset

    Reports:
        - ReportGenerator: Generate markdown/HTML/JSON reports
        - ReportConfig: Configuration for report generation
        - generate_markdown_report: Convenience function
        - save_report: Convenience function to save report

    HTML Reports:
        - HtmlReportGenerator: Generate styled, interactive HTML reports
        - HtmlReportConfig: Configuration for HTML reports
        - generate_html_report: Convenience function
        - save_html_report: Convenience function to save HTML report
"""

__version__ = "0.1.0"

from evaluation.metrics import (
    EvaluationMetrics,
    calculate_batch_metrics,
    calculate_metrics,
)

from evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkQuery,
    BenchmarkReport,
    BenchmarkRunner,
    StrategyResult,
    StrategyStatistics,
)

from evaluation.datasets import (
    Dataset,
    DatasetConfig,
    DatasetManager,
    DatasetQuery,
    DatasetSplit,
    create_dataset_from_queries,
    load_dataset,
)

from evaluation.reports import (
    ReportConfig,
    ReportGenerator,
    generate_markdown_report,
    save_report,
)

from evaluation.html_reports import (
    HtmlReportConfig,
    HtmlReportGenerator,
    generate_html_report,
    save_html_report,
)

__all__ = [
    "__version__",
    # Metrics
    "EvaluationMetrics",
    "calculate_batch_metrics",
    "calculate_metrics",
    # Benchmarks
    "BenchmarkConfig",
    "BenchmarkQuery",
    "BenchmarkReport",
    "BenchmarkRunner",
    "StrategyResult",
    "StrategyStatistics",
    # Datasets
    "Dataset",
    "DatasetConfig",
    "DatasetManager",
    "DatasetQuery",
    "DatasetSplit",
    "create_dataset_from_queries",
    "load_dataset",
    # Reports
    "ReportConfig",
    "ReportGenerator",
    "generate_markdown_report",
    "save_report",
    # HTML Reports
    "HtmlReportConfig",
    "HtmlReportGenerator",
    "generate_html_report",
    "save_html_report",
]
