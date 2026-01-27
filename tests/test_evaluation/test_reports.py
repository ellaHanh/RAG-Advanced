"""
Unit tests for Report Generation.

Tests cover:
- Markdown report generation
- HTML report generation
- JSON report generation
- Report configuration
- Saving reports to files
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkReport,
    StrategyStatistics,
)
from evaluation.reports import (
    ReportConfig,
    ReportGenerator,
    TableBuilder,
    generate_markdown_report,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_statistics() -> dict[str, StrategyStatistics]:
    """Create sample strategy statistics."""
    return {
        "standard": StrategyStatistics(
            strategy_name="standard",
            query_count=10,
            iteration_count=30,
            latency_p50=100.0,
            latency_p95=150.0,
            latency_p99=200.0,
            latency_mean=105.0,
            latency_std=25.0,
            total_cost=0.015,
            avg_cost_per_query=0.0015,
            success_rate=1.0,
            avg_precision={3: 0.8, 5: 0.7, 10: 0.6},
            avg_recall={3: 0.6, 5: 0.7, 10: 0.85},
            avg_mrr=0.85,
            avg_ndcg={3: 0.75, 5: 0.72, 10: 0.68},
        ),
        "reranking": StrategyStatistics(
            strategy_name="reranking",
            query_count=10,
            iteration_count=30,
            latency_p50=200.0,
            latency_p95=300.0,
            latency_p99=400.0,
            latency_mean=210.0,
            latency_std=50.0,
            total_cost=0.020,
            avg_cost_per_query=0.0020,
            success_rate=0.95,
            avg_precision={3: 0.9, 5: 0.85, 10: 0.75},
            avg_recall={3: 0.7, 5: 0.8, 10: 0.9},
            avg_mrr=0.92,
            avg_ndcg={3: 0.88, 5: 0.85, 10: 0.80},
        ),
    }


@pytest.fixture
def sample_report(sample_statistics: dict[str, StrategyStatistics]) -> BenchmarkReport:
    """Create a sample benchmark report."""
    config = BenchmarkConfig(
        strategies=["standard", "reranking"],
        iterations=3,
        k_values=[3, 5, 10],
    )

    return BenchmarkReport(
        benchmark_id="test-benchmark-123",
        config=config,
        statistics=sample_statistics,
        rankings={
            "latency_p50": ["standard", "reranking"],
            "latency_p95": ["standard", "reranking"],
            "mrr": ["reranking", "standard"],
            "precision@3": ["reranking", "standard"],
            "precision@5": ["reranking", "standard"],
            "ndcg@3": ["reranking", "standard"],
            "cost": ["standard", "reranking"],
        },
        duration_seconds=5.5,
        total_queries=10,
        total_executions=60,
    )


# =============================================================================
# Test: TableBuilder
# =============================================================================


class TestTableBuilder:
    """Tests for TableBuilder helper."""

    def test_basic_table(self):
        """Test basic table generation."""
        table = TableBuilder(
            headers=["Name", "Value"],
        )
        table.add_row("A", "1")
        table.add_row("B", "2")

        result = table.build()

        assert "| Name | Value |" in result
        assert "| A | 1 |" in result
        assert "| B | 2 |" in result

    def test_table_with_alignment(self):
        """Test table with alignment."""
        table = TableBuilder(
            headers=["Left", "Center", "Right"],
            alignment=["left", "center", "right"],
        )
        table.add_row("a", "b", "c")

        result = table.build()

        assert "| --- | :---: | ---: |" in result

    def test_empty_table(self):
        """Test empty table returns empty string."""
        table = TableBuilder()
        assert table.build() == ""


# =============================================================================
# Test: ReportConfig
# =============================================================================


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReportConfig()

        assert config.include_rankings is True
        assert config.include_cost_breakdown is True
        assert config.include_latency_stats is True
        assert config.include_metrics_tables is True
        assert config.decimal_places == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            include_rankings=False,
            decimal_places=2,
        )

        assert config.include_rankings is False
        assert config.decimal_places == 2


# =============================================================================
# Test: Markdown Generation
# =============================================================================


class TestMarkdownGeneration:
    """Tests for markdown report generation."""

    def test_generate_markdown_basic(self, sample_report: BenchmarkReport):
        """Test basic markdown generation."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report)

        assert "# RAG Strategy Benchmark Report" in markdown
        assert "## Overview" in markdown

    def test_generate_markdown_overview(self, sample_report: BenchmarkReport):
        """Test overview section."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report)

        assert "test-benchmark-123" in markdown
        assert "Total Queries**: 10" in markdown
        assert "Duration**: 5.50s" in markdown

    def test_generate_markdown_metrics_tables(self, sample_report: BenchmarkReport):
        """Test IR metrics tables."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report)

        assert "## IR Metrics" in markdown
        assert "### Precision@k" in markdown
        assert "### Recall@k" in markdown
        assert "### NDCG@k" in markdown
        assert "### Mean Reciprocal Rank" in markdown

    def test_generate_markdown_latency_section(self, sample_report: BenchmarkReport):
        """Test latency statistics section."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report)

        assert "## Latency Statistics" in markdown
        assert "P50 (ms)" in markdown
        assert "P95 (ms)" in markdown
        assert "P99 (ms)" in markdown

    def test_generate_markdown_cost_section(self, sample_report: BenchmarkReport):
        """Test cost analysis section."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report)

        assert "## Cost Analysis" in markdown
        assert "Total Cost ($)" in markdown
        assert "Avg Cost/Query" in markdown

    def test_generate_markdown_rankings_section(self, sample_report: BenchmarkReport):
        """Test rankings section."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report)

        assert "## Strategy Rankings" in markdown
        assert "By Latency" in markdown
        assert "By IR Metrics" in markdown
        assert "By Cost" in markdown

    def test_generate_markdown_without_rankings(self, sample_report: BenchmarkReport):
        """Test generation without rankings."""
        config = ReportConfig(include_rankings=False)
        generator = ReportGenerator(config)
        markdown = generator.generate_markdown(sample_report)

        assert "## Strategy Rankings" not in markdown

    def test_generate_markdown_without_cost(self, sample_report: BenchmarkReport):
        """Test generation without cost section."""
        config = ReportConfig(include_cost_breakdown=False)
        generator = ReportGenerator(config)
        markdown = generator.generate_markdown(sample_report)

        assert "## Cost Analysis" not in markdown

    def test_generate_markdown_custom_title(self, sample_report: BenchmarkReport):
        """Test custom report title."""
        generator = ReportGenerator()
        markdown = generator.generate_markdown(sample_report, title="Custom Report")

        assert "# Custom Report" in markdown


# =============================================================================
# Test: HTML Generation
# =============================================================================


class TestHtmlGeneration:
    """Tests for HTML report generation."""

    def test_generate_html_basic(self, sample_report: BenchmarkReport):
        """Test basic HTML generation."""
        generator = ReportGenerator()
        html = generator.generate_html(sample_report)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_generate_html_has_styles(self, sample_report: BenchmarkReport):
        """Test HTML includes styles."""
        generator = ReportGenerator()
        html = generator.generate_html(sample_report)

        assert "<style>" in html
        assert "font-family" in html

    def test_generate_html_has_title(self, sample_report: BenchmarkReport):
        """Test HTML has correct title."""
        generator = ReportGenerator()
        html = generator.generate_html(sample_report, title="Test Report")

        assert "<title>Test Report</title>" in html


# =============================================================================
# Test: JSON Generation
# =============================================================================


class TestJsonGeneration:
    """Tests for JSON report generation."""

    def test_generate_json_valid(self, sample_report: BenchmarkReport):
        """Test JSON is valid."""
        generator = ReportGenerator()
        json_str = generator.generate_json(sample_report)

        # Should be valid JSON
        data = json.loads(json_str)
        assert "benchmark_id" in data
        assert "statistics" in data
        assert "rankings" in data

    def test_generate_json_content(self, sample_report: BenchmarkReport):
        """Test JSON content."""
        generator = ReportGenerator()
        json_str = generator.generate_json(sample_report)
        data = json.loads(json_str)

        assert data["benchmark_id"] == "test-benchmark-123"
        assert "standard" in data["statistics"]
        assert data["total_queries"] == 10


# =============================================================================
# Test: Saving Reports
# =============================================================================


class TestSavingReports:
    """Tests for saving reports to files."""

    @pytest.mark.asyncio
    async def test_save_markdown(self, sample_report: BenchmarkReport):
        """Test saving markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            generator = ReportGenerator()
            await generator.save(sample_report, path)

            assert path.exists()
            content = path.read_text()
            assert "# RAG Strategy Benchmark Report" in content

    @pytest.mark.asyncio
    async def test_save_html(self, sample_report: BenchmarkReport):
        """Test saving HTML report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            generator = ReportGenerator()
            await generator.save(sample_report, path)

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    @pytest.mark.asyncio
    async def test_save_json(self, sample_report: BenchmarkReport):
        """Test saving JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            generator = ReportGenerator()
            await generator.save(sample_report, path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert "benchmark_id" in data

    def test_save_sync(self, sample_report: BenchmarkReport):
        """Test synchronous save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            generator = ReportGenerator()
            generator.save_sync(sample_report, path)

            assert path.exists()

    @pytest.mark.asyncio
    async def test_save_creates_directories(self, sample_report: BenchmarkReport):
        """Test save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "report.md"
            generator = ReportGenerator()
            await generator.save(sample_report, path)

            assert path.exists()


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_markdown_report(self, sample_report: BenchmarkReport):
        """Test convenience function."""
        markdown = generate_markdown_report(sample_report)

        assert "# RAG Strategy Benchmark Report" in markdown
        assert "## Overview" in markdown
