"""
RAG-Advanced Report Generation.

Generate markdown, HTML, and JSON evaluation reports with metrics tables,
strategy rankings, and cost breakdowns.

Usage:
    from evaluation.reports import ReportGenerator, ReportConfig

    generator = ReportGenerator()
    markdown = generator.generate_markdown(benchmark_report)
    await generator.save(benchmark_report, "reports/benchmark.md")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, ConfigDict, Field

from evaluation.benchmarks import BenchmarkReport, StrategyStatistics


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class ReportConfig(BaseModel):
    """
    Configuration for report generation.

    Attributes:
        include_rankings: Whether to include strategy rankings.
        include_cost_breakdown: Whether to include cost analysis.
        include_latency_stats: Whether to include latency statistics.
        include_metrics_tables: Whether to include IR metrics tables.
        decimal_places: Number of decimal places for metrics.
        include_timestamp: Whether to include generation timestamp.
    """

    model_config = ConfigDict(frozen=True)

    include_rankings: bool = Field(default=True, description="Include rankings")
    include_cost_breakdown: bool = Field(default=True, description="Include cost analysis")
    include_latency_stats: bool = Field(default=True, description="Include latency stats")
    include_metrics_tables: bool = Field(default=True, description="Include metrics tables")
    decimal_places: int = Field(default=3, ge=1, le=6, description="Decimal places")
    include_timestamp: bool = Field(default=True, description="Include timestamp")


# =============================================================================
# Table Builder Helper
# =============================================================================


@dataclass
class TableBuilder:
    """Helper for building markdown tables."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    alignment: list[str] = field(default_factory=list)

    def add_row(self, *values: Any) -> None:
        """Add a row to the table."""
        self.rows.append([str(v) for v in values])

    def build(self) -> str:
        """Build the markdown table."""
        if not self.headers:
            return ""

        lines = []

        # Header row
        lines.append("| " + " | ".join(self.headers) + " |")

        # Separator with alignment
        sep_parts = []
        for i, h in enumerate(self.headers):
            align = self.alignment[i] if i < len(self.alignment) else "left"
            if align == "center":
                sep_parts.append(":---:")
            elif align == "right":
                sep_parts.append("---:")
            else:
                sep_parts.append("---")
        lines.append("| " + " | ".join(sep_parts) + " |")

        # Data rows
        for row in self.rows:
            # Pad row if needed
            while len(row) < len(self.headers):
                row.append("")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """
    Generate evaluation reports in various formats.

    Supports markdown, HTML, and JSON output formats with customizable sections.

    Example:
        >>> generator = ReportGenerator()
        >>> markdown = generator.generate_markdown(benchmark_report)
        >>> await generator.save(benchmark_report, "report.md")
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """
        Initialize the report generator.

        Args:
            config: Optional report configuration.
        """
        self.config = config or ReportConfig()

    def generate_markdown(
        self,
        report: BenchmarkReport,
        title: str = "RAG Strategy Benchmark Report",
    ) -> str:
        """
        Generate a markdown report.

        Args:
            report: Benchmark report to format.
            title: Report title.

        Returns:
            Markdown-formatted report string.
        """
        sections = []

        # Title
        sections.append(f"# {title}\n")

        # Overview section
        sections.append(self._generate_overview(report))

        # Metrics tables
        if self.config.include_metrics_tables:
            sections.append(self._generate_metrics_tables(report))

        # Latency statistics
        if self.config.include_latency_stats:
            sections.append(self._generate_latency_section(report))

        # Cost breakdown
        if self.config.include_cost_breakdown:
            sections.append(self._generate_cost_section(report))

        # Rankings
        if self.config.include_rankings:
            sections.append(self._generate_rankings_section(report))

        # Footer with timestamp
        if self.config.include_timestamp:
            sections.append(self._generate_footer())

        return "\n".join(filter(None, sections))

    def generate_html(
        self,
        report: BenchmarkReport,
        title: str = "RAG Strategy Benchmark Report",
    ) -> str:
        """
        Generate an HTML report.

        Args:
            report: Benchmark report to format.
            title: Report title.

        Returns:
            HTML-formatted report string.
        """
        markdown = self.generate_markdown(report, title)

        # Basic HTML wrapper with styling
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 0.75rem;
            text-align: left;
        }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{
            background-color: #f4f4f4;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .best {{ color: #27ae60; font-weight: bold; }}
        .footer {{ color: #666; font-size: 0.9em; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; }}
    </style>
</head>
<body>
<pre>{self._markdown_to_html_basic(markdown)}</pre>
</body>
</html>"""
        return html

    def generate_json(
        self,
        report: BenchmarkReport,
    ) -> str:
        """
        Generate a JSON report.

        Args:
            report: Benchmark report to format.

        Returns:
            JSON-formatted report string.
        """
        return json.dumps(report.to_dict(), indent=2, default=str)

    async def save(
        self,
        report: BenchmarkReport,
        path: Path | str,
        format: str | None = None,
    ) -> None:
        """
        Save report to file.

        Args:
            report: Benchmark report to save.
            path: Output file path.
            format: Output format ('markdown', 'html', 'json'). Auto-detected from extension.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect format from extension
        if format is None:
            ext = path.suffix.lower()
            if ext in (".html", ".htm"):
                format = "html"
            elif ext == ".json":
                format = "json"
            else:
                format = "markdown"

        if format == "html":
            content = self.generate_html(report)
        elif format == "json":
            content = self.generate_json(report)
        else:
            content = self.generate_markdown(report)

        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.info(f"Saved report to {path}")

    def save_sync(
        self,
        report: BenchmarkReport,
        path: Path | str,
        format: str | None = None,
    ) -> None:
        """
        Synchronous version of save.

        Args:
            report: Benchmark report to save.
            path: Output file path.
            format: Output format.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format is None:
            ext = path.suffix.lower()
            if ext in (".html", ".htm"):
                format = "html"
            elif ext == ".json":
                format = "json"
            else:
                format = "markdown"

        if format == "html":
            content = self.generate_html(report)
        elif format == "json":
            content = self.generate_json(report)
        else:
            content = self.generate_markdown(report)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # =========================================================================
    # Private Methods - Section Generators
    # =========================================================================

    def _generate_overview(self, report: BenchmarkReport) -> str:
        """Generate the overview section."""
        lines = [
            "## Overview\n",
            f"- **Benchmark ID**: `{report.benchmark_id}`",
            f"- **Total Queries**: {report.total_queries}",
            f"- **Total Executions**: {report.total_executions}",
            f"- **Duration**: {report.duration_seconds:.2f}s",
            f"- **Strategies**: {', '.join(report.config.strategies)}",
            f"- **Iterations**: {report.config.iterations}",
            "",
        ]
        return "\n".join(lines)

    def _generate_metrics_tables(self, report: BenchmarkReport) -> str:
        """Generate IR metrics tables."""
        if not report.statistics:
            return ""

        dp = self.config.decimal_places
        lines = ["## IR Metrics\n"]

        # Precision table
        lines.append("### Precision@k\n")
        precision_table = TableBuilder(
            headers=["Strategy"] + [f"P@{k}" for k in report.config.k_values],
            alignment=["left"] + ["right"] * len(report.config.k_values),
        )
        for name, stats in report.statistics.items():
            row = [name] + [f"{stats.avg_precision.get(k, 0.0):.{dp}f}" for k in report.config.k_values]
            precision_table.add_row(*row)
        lines.append(precision_table.build())
        lines.append("")

        # Recall table
        lines.append("### Recall@k\n")
        recall_table = TableBuilder(
            headers=["Strategy"] + [f"R@{k}" for k in report.config.k_values],
            alignment=["left"] + ["right"] * len(report.config.k_values),
        )
        for name, stats in report.statistics.items():
            row = [name] + [f"{stats.avg_recall.get(k, 0.0):.{dp}f}" for k in report.config.k_values]
            recall_table.add_row(*row)
        lines.append(recall_table.build())
        lines.append("")

        # NDCG table
        lines.append("### NDCG@k\n")
        ndcg_table = TableBuilder(
            headers=["Strategy"] + [f"NDCG@{k}" for k in report.config.k_values],
            alignment=["left"] + ["right"] * len(report.config.k_values),
        )
        for name, stats in report.statistics.items():
            row = [name] + [f"{stats.avg_ndcg.get(k, 0.0):.{dp}f}" for k in report.config.k_values]
            ndcg_table.add_row(*row)
        lines.append(ndcg_table.build())
        lines.append("")

        # MRR table
        lines.append("### Mean Reciprocal Rank (MRR)\n")
        mrr_table = TableBuilder(
            headers=["Strategy", "MRR"],
            alignment=["left", "right"],
        )
        for name, stats in report.statistics.items():
            mrr_table.add_row(name, f"{stats.avg_mrr:.{dp}f}")
        lines.append(mrr_table.build())
        lines.append("")

        return "\n".join(lines)

    def _generate_latency_section(self, report: BenchmarkReport) -> str:
        """Generate latency statistics section."""
        if not report.statistics:
            return ""

        lines = ["## Latency Statistics\n"]

        latency_table = TableBuilder(
            headers=["Strategy", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Mean (ms)", "Std Dev"],
            alignment=["left", "right", "right", "right", "right", "right"],
        )

        for name, stats in report.statistics.items():
            latency_table.add_row(
                name,
                f"{stats.latency_p50:.0f}",
                f"{stats.latency_p95:.0f}",
                f"{stats.latency_p99:.0f}",
                f"{stats.latency_mean:.0f}",
                f"{stats.latency_std:.1f}",
            )

        lines.append(latency_table.build())
        lines.append("")

        return "\n".join(lines)

    def _generate_cost_section(self, report: BenchmarkReport) -> str:
        """Generate cost breakdown section."""
        if not report.statistics:
            return ""

        dp = self.config.decimal_places
        lines = ["## Cost Analysis\n"]

        cost_table = TableBuilder(
            headers=["Strategy", "Total Cost ($)", "Avg Cost/Query ($)", "Success Rate"],
            alignment=["left", "right", "right", "right"],
        )

        for name, stats in report.statistics.items():
            cost_table.add_row(
                name,
                f"{stats.total_cost:.4f}",
                f"{stats.avg_cost_per_query:.{dp+1}f}",
                f"{stats.success_rate * 100:.1f}%",
            )

        lines.append(cost_table.build())
        lines.append("")

        # Total cost summary
        total_cost = sum(s.total_cost for s in report.statistics.values())
        lines.append(f"**Total Cost**: ${total_cost:.4f}\n")

        return "\n".join(lines)

    def _generate_rankings_section(self, report: BenchmarkReport) -> str:
        """Generate strategy rankings section."""
        if not report.rankings:
            return ""

        lines = ["## Strategy Rankings\n"]

        # Group rankings by category
        latency_rankings = {k: v for k, v in report.rankings.items() if "latency" in k}
        metric_rankings = {k: v for k, v in report.rankings.items()
                         if k not in latency_rankings and k != "cost"}
        cost_ranking = report.rankings.get("cost", [])

        if latency_rankings:
            lines.append("### By Latency (Lower is Better)\n")
            for metric, ranking in latency_rankings.items():
                lines.append(f"- **{metric}**: {' → '.join(ranking)}")
            lines.append("")

        if metric_rankings:
            lines.append("### By IR Metrics (Higher is Better)\n")
            for metric, ranking in metric_rankings.items():
                lines.append(f"- **{metric}**: {' → '.join(ranking)}")
            lines.append("")

        if cost_ranking:
            lines.append("### By Cost (Lower is Better)\n")
            lines.append(f"- **cost**: {' → '.join(cost_ranking)}")
            lines.append("")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate report footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"---\n*Report generated on {timestamp}*\n"

    def _markdown_to_html_basic(self, markdown: str) -> str:
        """Basic markdown to HTML conversion for the pre block."""
        # This is a simple passthrough - for proper conversion, use a library
        return markdown


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_markdown_report(
    report: BenchmarkReport,
    title: str = "RAG Strategy Benchmark Report",
) -> str:
    """
    Generate a markdown report.

    Args:
        report: Benchmark report.
        title: Report title.

    Returns:
        Markdown string.
    """
    generator = ReportGenerator()
    return generator.generate_markdown(report, title)


async def save_report(
    report: BenchmarkReport,
    path: Path | str,
    format: str | None = None,
) -> None:
    """
    Save a benchmark report to file.

    Args:
        report: Benchmark report.
        path: Output file path.
        format: Output format (auto-detected from extension).
    """
    generator = ReportGenerator()
    await generator.save(report, path, format)
