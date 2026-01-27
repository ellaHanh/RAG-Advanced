"""
Unit tests for HTML Report Generation.

Tests cover:
- HTML report generation
- Styling and theming
- Interactive elements
- Configuration options
- File saving
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evaluation.benchmarks import (
    BenchmarkConfig,
    BenchmarkReport,
    StrategyStatistics,
)
from evaluation.html_reports import (
    HtmlReportConfig,
    HtmlReportGenerator,
    generate_html_report,
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
            avg_precision={3: 0.8, 5: 0.7},
            avg_recall={3: 0.6, 5: 0.7},
            avg_mrr=0.85,
            avg_ndcg={3: 0.75, 5: 0.72},
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
            avg_precision={3: 0.9, 5: 0.85},
            avg_recall={3: 0.7, 5: 0.8},
            avg_mrr=0.92,
            avg_ndcg={3: 0.88, 5: 0.85},
        ),
    }


@pytest.fixture
def sample_report(sample_statistics: dict[str, StrategyStatistics]) -> BenchmarkReport:
    """Create a sample benchmark report."""
    config = BenchmarkConfig(
        strategies=["standard", "reranking"],
        iterations=3,
        k_values=[3, 5],
    )

    return BenchmarkReport(
        benchmark_id="test-benchmark-123",
        config=config,
        statistics=sample_statistics,
        rankings={
            "latency_p50": ["standard", "reranking"],
            "mrr": ["reranking", "standard"],
            "cost": ["standard", "reranking"],
        },
        duration_seconds=5.5,
        total_queries=10,
        total_executions=60,
    )


# =============================================================================
# Test: HtmlReportConfig
# =============================================================================


class TestHtmlReportConfig:
    """Tests for HtmlReportConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HtmlReportConfig()

        assert config.theme == "light"
        assert config.include_charts is True
        assert config.include_raw_data is False
        assert config.sortable_tables is True
        assert config.decimal_places == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = HtmlReportConfig(
            theme="dark",
            include_charts=False,
            include_raw_data=True,
            company_name="Test Corp",
        )

        assert config.theme == "dark"
        assert config.include_charts is False
        assert config.include_raw_data is True
        assert config.company_name == "Test Corp"


# =============================================================================
# Test: HTML Generation - Structure
# =============================================================================


class TestHtmlGenerationStructure:
    """Tests for HTML report structure."""

    def test_valid_html(self, sample_report: BenchmarkReport):
        """Test that generated HTML is valid."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_has_title(self, sample_report: BenchmarkReport):
        """Test that HTML has correct title."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report, title="Custom Title")

        assert "<title>Custom Title</title>" in html

    def test_has_styles(self, sample_report: BenchmarkReport):
        """Test that HTML includes styles."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "<style>" in html
        assert ":root {" in html
        assert "font-family" in html

    def test_has_javascript(self, sample_report: BenchmarkReport):
        """Test that HTML includes JavaScript."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "<script>" in html
        assert "sortTable" in html


# =============================================================================
# Test: HTML Generation - Theming
# =============================================================================


class TestHtmlGenerationTheming:
    """Tests for HTML theming."""

    def test_light_theme(self, sample_report: BenchmarkReport):
        """Test light theme."""
        config = HtmlReportConfig(theme="light")
        generator = HtmlReportGenerator(config)
        html = generator.generate(sample_report)

        assert "--bg-primary: #ffffff" in html

    def test_dark_theme(self, sample_report: BenchmarkReport):
        """Test dark theme."""
        config = HtmlReportConfig(theme="dark")
        generator = HtmlReportGenerator(config)
        html = generator.generate(sample_report)

        assert "--bg-primary: #1a1a2e" in html


# =============================================================================
# Test: HTML Generation - Sections
# =============================================================================


class TestHtmlGenerationSections:
    """Tests for HTML report sections."""

    def test_has_header(self, sample_report: BenchmarkReport):
        """Test that report has header."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "<header>" in html
        assert "test-benchmark-123" in html

    def test_has_overview(self, sample_report: BenchmarkReport):
        """Test that report has overview."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "Overview" in html
        assert "Total Queries" in html
        assert "10" in html  # total_queries

    def test_has_metrics_section(self, sample_report: BenchmarkReport):
        """Test that report has metrics section."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "IR Metrics" in html
        assert "Precision@k" in html
        assert "NDCG@k" in html
        assert "MRR" in html

    def test_has_latency_section(self, sample_report: BenchmarkReport):
        """Test that report has latency section."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "Latency Statistics" in html
        assert "P50 (ms)" in html
        assert "P95 (ms)" in html

    def test_has_cost_section(self, sample_report: BenchmarkReport):
        """Test that report has cost section."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "Cost Analysis" in html
        assert "Total Cost" in html

    def test_has_rankings_section(self, sample_report: BenchmarkReport):
        """Test that report has rankings section."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "Rankings" in html
        assert "latency_p50" in html

    def test_has_footer(self, sample_report: BenchmarkReport):
        """Test that report has footer."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "<footer>" in html
        assert "Report generated on" in html


# =============================================================================
# Test: HTML Generation - Optional Sections
# =============================================================================


class TestHtmlGenerationOptionalSections:
    """Tests for optional HTML sections."""

    def test_includes_charts_by_default(self, sample_report: BenchmarkReport):
        """Test that charts are included by default."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "Visualizations" in html
        assert "chart-placeholder" in html

    def test_excludes_charts_when_disabled(self, sample_report: BenchmarkReport):
        """Test that charts can be disabled."""
        config = HtmlReportConfig(include_charts=False)
        generator = HtmlReportGenerator(config)
        html = generator.generate(sample_report)

        # Charts section content should not be present (CSS class will still be in styles)
        assert "Visualizations" not in html
        assert "Latency Comparison Chart" not in html

    def test_excludes_raw_data_by_default(self, sample_report: BenchmarkReport):
        """Test that raw data is excluded by default."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert "Raw Data" not in html

    def test_includes_raw_data_when_enabled(self, sample_report: BenchmarkReport):
        """Test that raw data can be included."""
        config = HtmlReportConfig(include_raw_data=True)
        generator = HtmlReportGenerator(config)
        html = generator.generate(sample_report)

        assert "Raw Data" in html
        assert "benchmark_id" in html


# =============================================================================
# Test: HTML Generation - Interactive Elements
# =============================================================================


class TestHtmlGenerationInteractive:
    """Tests for interactive HTML elements."""

    def test_sortable_tables_enabled(self, sample_report: BenchmarkReport):
        """Test sortable tables are included."""
        config = HtmlReportConfig(sortable_tables=True)
        generator = HtmlReportGenerator(config)
        html = generator.generate(sample_report)

        assert 'class="sortable"' in html

    def test_sortable_tables_disabled(self, sample_report: BenchmarkReport):
        """Test sortable tables can be disabled."""
        config = HtmlReportConfig(sortable_tables=False)
        generator = HtmlReportGenerator(config)
        html = generator.generate(sample_report)

        assert 'class="sortable"' not in html

    def test_has_tabs(self, sample_report: BenchmarkReport):
        """Test that metrics section has tabs."""
        generator = HtmlReportGenerator()
        html = generator.generate(sample_report)

        assert 'class="tab' in html
        assert 'data-target=' in html


# =============================================================================
# Test: File Saving
# =============================================================================


class TestHtmlFileSaving:
    """Tests for saving HTML reports."""

    @pytest.mark.asyncio
    async def test_save_async(self, sample_report: BenchmarkReport):
        """Test async file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            generator = HtmlReportGenerator()
            await generator.save(sample_report, path)

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_save_sync(self, sample_report: BenchmarkReport):
        """Test sync file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            generator = HtmlReportGenerator()
            generator.save_sync(sample_report, path)

            assert path.exists()

    @pytest.mark.asyncio
    async def test_save_creates_directories(self, sample_report: BenchmarkReport):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "report.html"
            generator = HtmlReportGenerator()
            await generator.save(sample_report, path)

            assert path.exists()


# =============================================================================
# Test: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_html_report(self, sample_report: BenchmarkReport):
        """Test convenience function."""
        html = generate_html_report(sample_report)

        assert "<!DOCTYPE html>" in html
        assert "Overview" in html

    def test_generate_html_report_with_config(self, sample_report: BenchmarkReport):
        """Test convenience function with config."""
        config = HtmlReportConfig(theme="dark")
        html = generate_html_report(sample_report, config=config)

        assert "--bg-primary: #1a1a2e" in html
