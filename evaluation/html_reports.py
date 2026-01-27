"""
RAG-Advanced HTML Report Generator.

Generate styled, interactive HTML evaluation reports with visual elements,
sortable tables, and chart placeholders for visualization libraries.

Usage:
    from evaluation.html_reports import HtmlReportGenerator

    generator = HtmlReportGenerator()
    html = generator.generate(benchmark_report)
    await generator.save(benchmark_report, "reports/benchmark.html")
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass
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


class HtmlReportConfig(BaseModel):
    """
    Configuration for HTML report generation.

    Attributes:
        theme: Color theme ('light' or 'dark').
        include_charts: Whether to include chart placeholders.
        include_raw_data: Whether to include raw JSON data.
        sortable_tables: Whether tables should be sortable.
        decimal_places: Number of decimal places for metrics.
        company_name: Optional company name for branding.
    """

    model_config = ConfigDict(frozen=True)

    theme: str = Field(default="light", description="Color theme")
    include_charts: bool = Field(default=True, description="Include chart placeholders")
    include_raw_data: bool = Field(default=False, description="Include raw JSON data")
    sortable_tables: bool = Field(default=True, description="Sortable tables")
    decimal_places: int = Field(default=3, ge=1, le=6, description="Decimal places")
    company_name: str | None = Field(default=None, description="Company name")


# =============================================================================
# CSS Styles
# =============================================================================


LIGHT_THEME_CSS = """
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-tertiary: #e9ecef;
    --text-primary: #212529;
    --text-secondary: #6c757d;
    --text-muted: #adb5bd;
    --border-color: #dee2e6;
    --accent-primary: #0d6efd;
    --accent-success: #198754;
    --accent-warning: #ffc107;
    --accent-danger: #dc3545;
    --shadow: rgba(0, 0, 0, 0.1);
}
"""

DARK_THEME_CSS = """
:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --bg-tertiary: #0f3460;
    --text-primary: #eaeaea;
    --text-secondary: #b3b3b3;
    --text-muted: #6c6c6c;
    --border-color: #3a3a5a;
    --accent-primary: #4cc9f0;
    --accent-success: #4ade80;
    --accent-warning: #facc15;
    --accent-danger: #f87171;
    --shadow: rgba(0, 0, 0, 0.3);
}
"""

BASE_CSS = """
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #6610f2 100%);
    color: white;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    border-radius: 0 0 1rem 1rem;
}

header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

header .subtitle {
    opacity: 0.9;
    font-size: 1.1rem;
}

.card {
    background-color: var(--bg-secondary);
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px var(--shadow);
    border: 1px solid var(--border-color);
}

.card h2 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--accent-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.stat-card {
    background-color: var(--bg-tertiary);
    padding: 1.25rem;
    border-radius: 0.5rem;
    text-align: center;
}

.stat-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-primary);
}

.stat-card .label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

th {
    background-color: var(--bg-tertiary);
    font-weight: 600;
    color: var(--text-primary);
    position: sticky;
    top: 0;
}

th.sortable {
    cursor: pointer;
    user-select: none;
}

th.sortable:hover {
    background-color: var(--accent-primary);
    color: white;
}

th.sortable::after {
    content: ' ↕';
    opacity: 0.5;
}

th.sorted-asc::after {
    content: ' ↑';
    opacity: 1;
}

th.sorted-desc::after {
    content: ' ↓';
    opacity: 1;
}

tr:hover {
    background-color: var(--bg-tertiary);
}

.best {
    color: var(--accent-success);
    font-weight: 600;
}

.worst {
    color: var(--accent-danger);
}

.chart-placeholder {
    background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
    border: 2px dashed var(--border-color);
    border-radius: 0.5rem;
    padding: 3rem;
    text-align: center;
    color: var(--text-muted);
    margin: 1rem 0;
}

.chart-placeholder svg {
    width: 48px;
    height: 48px;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.ranking-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.ranking-item {
    background-color: var(--bg-tertiary);
    padding: 0.25rem 0.75rem;
    border-radius: 2rem;
    font-size: 0.875rem;
}

.ranking-item:first-child {
    background-color: var(--accent-success);
    color: white;
}

.ranking-item::before {
    content: counter(rank) ". ";
    counter-increment: rank;
    font-weight: 600;
}

.rankings-section {
    counter-reset: rank;
}

footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-muted);
    font-size: 0.875rem;
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
}

.tabs {
    display: flex;
    gap: 0.25rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-color);
}

.tab {
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    border: none;
    background: none;
    color: var(--text-secondary);
    font-size: 0.9rem;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}

.tab:hover {
    color: var(--accent-primary);
}

.tab.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
    font-weight: 600;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

@media (max-width: 768px) {
    .container {
        padding: 1rem;
    }
    
    header {
        padding: 2rem 1rem;
    }
    
    header h1 {
        font-size: 1.75rem;
    }
    
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    table {
        font-size: 0.8rem;
    }
    
    th, td {
        padding: 0.5rem;
    }
}
"""

JAVASCRIPT = """
// Table sorting
function sortTable(table, column, asc = true) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aVal = a.cells[column].textContent;
        const bVal = b.cells[column].textContent;
        
        const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
        const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return asc ? aNum - bNum : bNum - aNum;
        }
        
        return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

document.querySelectorAll('th.sortable').forEach((th, index) => {
    th.addEventListener('click', function() {
        const table = this.closest('table');
        const isAsc = this.classList.contains('sorted-asc');
        
        table.querySelectorAll('th').forEach(h => {
            h.classList.remove('sorted-asc', 'sorted-desc');
        });
        
        this.classList.add(isAsc ? 'sorted-desc' : 'sorted-asc');
        sortTable(table, index, !isAsc);
    });
});

// Tab functionality
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function() {
        const tabGroup = this.closest('.card');
        const targetId = this.dataset.target;
        
        tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        this.classList.add('active');
        document.getElementById(targetId).classList.add('active');
    });
});

// Highlight best/worst values in tables
document.querySelectorAll('table').forEach(table => {
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    const cols = table.querySelectorAll('th').length;
    
    for (let col = 1; col < cols; col++) {
        const cells = Array.from(tbody.querySelectorAll(`tr td:nth-child(${col + 1})`));
        const values = cells.map(c => parseFloat(c.textContent.replace(/[^0-9.-]/g, '')));
        
        if (values.every(v => !isNaN(v))) {
            const max = Math.max(...values);
            const min = Math.min(...values);
            
            cells.forEach((cell, i) => {
                if (values[i] === max) cell.classList.add('best');
                if (values[i] === min && min !== max) cell.classList.add('worst');
            });
        }
    }
});
"""


# =============================================================================
# HTML Report Generator
# =============================================================================


class HtmlReportGenerator:
    """
    Generate styled, interactive HTML evaluation reports.

    Example:
        >>> generator = HtmlReportGenerator()
        >>> html = generator.generate(benchmark_report)
    """

    def __init__(self, config: HtmlReportConfig | None = None) -> None:
        """
        Initialize the generator.

        Args:
            config: Optional configuration.
        """
        self.config = config or HtmlReportConfig()

    def generate(
        self,
        report: BenchmarkReport,
        title: str = "RAG Strategy Benchmark Report",
    ) -> str:
        """
        Generate an HTML report.

        Args:
            report: Benchmark report.
            title: Report title.

        Returns:
            HTML string.
        """
        theme_css = DARK_THEME_CSS if self.config.theme == "dark" else LIGHT_THEME_CSS

        sections = [
            self._generate_header(report, title),
            self._generate_overview_cards(report),
            self._generate_metrics_section(report),
            self._generate_latency_section(report),
            self._generate_cost_section(report),
            self._generate_rankings_section(report),
        ]

        if self.config.include_charts:
            sections.insert(2, self._generate_charts_section(report))

        if self.config.include_raw_data:
            sections.append(self._generate_raw_data_section(report))

        body_content = "\n".join(sections)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        {theme_css}
        {BASE_CSS}
    </style>
</head>
<body>
    {body_content}
    {self._generate_footer()}
    <script>
        {JAVASCRIPT}
    </script>
</body>
</html>"""

    async def save(
        self,
        report: BenchmarkReport,
        path: Path | str,
        title: str = "RAG Strategy Benchmark Report",
    ) -> None:
        """
        Save report to HTML file.

        Args:
            report: Benchmark report.
            path: Output file path.
            title: Report title.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate(report, title)

        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

        logger.info(f"Saved HTML report to {path}")

    def save_sync(
        self,
        report: BenchmarkReport,
        path: Path | str,
        title: str = "RAG Strategy Benchmark Report",
    ) -> None:
        """
        Synchronous save.

        Args:
            report: Benchmark report.
            path: Output file path.
            title: Report title.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.generate(report, title)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _generate_header(self, report: BenchmarkReport, title: str) -> str:
        """Generate header section."""
        subtitle = f"Benchmark ID: {report.benchmark_id}"
        company = f" | {self.config.company_name}" if self.config.company_name else ""

        return f"""
<header>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        <div class="subtitle">{html.escape(subtitle)}{company}</div>
    </div>
</header>
<main class="container">
"""

    def _generate_overview_cards(self, report: BenchmarkReport) -> str:
        """Generate overview statistics cards."""
        return f"""
<div class="card">
    <h2>📊 Overview</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{report.total_queries}</div>
            <div class="label">Total Queries</div>
        </div>
        <div class="stat-card">
            <div class="value">{report.total_executions}</div>
            <div class="label">Total Executions</div>
        </div>
        <div class="stat-card">
            <div class="value">{report.duration_seconds:.2f}s</div>
            <div class="label">Duration</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(report.config.strategies)}</div>
            <div class="label">Strategies</div>
        </div>
    </div>
</div>
"""

    def _generate_metrics_section(self, report: BenchmarkReport) -> str:
        """Generate IR metrics section with tabbed tables."""
        if not report.statistics:
            return ""

        dp = self.config.decimal_places
        sortable = "sortable" if self.config.sortable_tables else ""

        # Build precision table
        precision_rows = ""
        for name, stats in report.statistics.items():
            cells = "".join(
                f"<td>{stats.avg_precision.get(k, 0.0):.{dp}f}</td>"
                for k in report.config.k_values
            )
            precision_rows += f"<tr><td>{html.escape(name)}</td>{cells}</tr>"

        precision_headers = "".join(f"<th class='{sortable}'>P@{k}</th>" for k in report.config.k_values)

        # Build NDCG table
        ndcg_rows = ""
        for name, stats in report.statistics.items():
            cells = "".join(
                f"<td>{stats.avg_ndcg.get(k, 0.0):.{dp}f}</td>"
                for k in report.config.k_values
            )
            ndcg_rows += f"<tr><td>{html.escape(name)}</td>{cells}</tr>"

        ndcg_headers = "".join(f"<th class='{sortable}'>NDCG@{k}</th>" for k in report.config.k_values)

        # Build MRR table
        mrr_rows = ""
        for name, stats in report.statistics.items():
            mrr_rows += f"<tr><td>{html.escape(name)}</td><td>{stats.avg_mrr:.{dp}f}</td></tr>"

        return f"""
<div class="card">
    <h2>📈 IR Metrics</h2>
    <div class="tabs">
        <button class="tab active" data-target="precision-tab">Precision@k</button>
        <button class="tab" data-target="ndcg-tab">NDCG@k</button>
        <button class="tab" data-target="mrr-tab">MRR</button>
    </div>
    
    <div id="precision-tab" class="tab-content active">
        <table>
            <thead>
                <tr>
                    <th class="{sortable}">Strategy</th>
                    {precision_headers}
                </tr>
            </thead>
            <tbody>{precision_rows}</tbody>
        </table>
    </div>
    
    <div id="ndcg-tab" class="tab-content">
        <table>
            <thead>
                <tr>
                    <th class="{sortable}">Strategy</th>
                    {ndcg_headers}
                </tr>
            </thead>
            <tbody>{ndcg_rows}</tbody>
        </table>
    </div>
    
    <div id="mrr-tab" class="tab-content">
        <table>
            <thead>
                <tr>
                    <th class="{sortable}">Strategy</th>
                    <th class="{sortable}">MRR</th>
                </tr>
            </thead>
            <tbody>{mrr_rows}</tbody>
        </table>
    </div>
</div>
"""

    def _generate_latency_section(self, report: BenchmarkReport) -> str:
        """Generate latency statistics section."""
        if not report.statistics:
            return ""

        sortable = "sortable" if self.config.sortable_tables else ""

        rows = ""
        for name, stats in report.statistics.items():
            rows += f"""
<tr>
    <td>{html.escape(name)}</td>
    <td>{stats.latency_p50:.0f}</td>
    <td>{stats.latency_p95:.0f}</td>
    <td>{stats.latency_p99:.0f}</td>
    <td>{stats.latency_mean:.0f}</td>
    <td>{stats.latency_std:.1f}</td>
</tr>
"""

        return f"""
<div class="card">
    <h2>⏱️ Latency Statistics</h2>
    <table>
        <thead>
            <tr>
                <th class="{sortable}">Strategy</th>
                <th class="{sortable}">P50 (ms)</th>
                <th class="{sortable}">P95 (ms)</th>
                <th class="{sortable}">P99 (ms)</th>
                <th class="{sortable}">Mean (ms)</th>
                <th class="{sortable}">Std Dev</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
</div>
"""

    def _generate_cost_section(self, report: BenchmarkReport) -> str:
        """Generate cost analysis section."""
        if not report.statistics:
            return ""

        dp = self.config.decimal_places
        sortable = "sortable" if self.config.sortable_tables else ""
        total_cost = sum(s.total_cost for s in report.statistics.values())

        rows = ""
        for name, stats in report.statistics.items():
            rows += f"""
<tr>
    <td>{html.escape(name)}</td>
    <td>${stats.total_cost:.4f}</td>
    <td>${stats.avg_cost_per_query:.{dp+1}f}</td>
    <td>{stats.success_rate * 100:.1f}%</td>
</tr>
"""

        return f"""
<div class="card">
    <h2>💰 Cost Analysis</h2>
    <table>
        <thead>
            <tr>
                <th class="{sortable}">Strategy</th>
                <th class="{sortable}">Total Cost ($)</th>
                <th class="{sortable}">Avg Cost/Query ($)</th>
                <th class="{sortable}">Success Rate</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    <p style="margin-top: 1rem; color: var(--text-secondary);">
        <strong>Total Benchmark Cost:</strong> ${total_cost:.4f}
    </p>
</div>
"""

    def _generate_rankings_section(self, report: BenchmarkReport) -> str:
        """Generate strategy rankings section."""
        if not report.rankings:
            return ""

        ranking_items = ""
        for metric, strategies in report.rankings.items():
            items = "".join(
                f'<span class="ranking-item">{html.escape(s)}</span>'
                for s in strategies
            )
            ranking_items += f"""
<div style="margin-bottom: 1rem;">
    <strong style="color: var(--text-secondary);">{html.escape(metric)}:</strong>
    <div class="ranking-list" style="counter-reset: rank;">{items}</div>
</div>
"""

        return f"""
<div class="card rankings-section">
    <h2>🏆 Strategy Rankings</h2>
    {ranking_items}
</div>
"""

    def _generate_charts_section(self, report: BenchmarkReport) -> str:
        """Generate chart placeholders section."""
        return """
<div class="card">
    <h2>📊 Visualizations</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
        <div class="chart-placeholder">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 20V10M12 20V4M6 20v-6"/>
            </svg>
            <div>Latency Comparison Chart</div>
            <small>Add Chart.js or D3.js for visualization</small>
        </div>
        <div class="chart-placeholder">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 2v10l4.5 4.5"/>
            </svg>
            <div>Metrics Radar Chart</div>
            <small>Add Chart.js or D3.js for visualization</small>
        </div>
    </div>
</div>
"""

    def _generate_raw_data_section(self, report: BenchmarkReport) -> str:
        """Generate raw JSON data section."""
        json_data = json.dumps(report.to_dict(), indent=2, default=str)

        return f"""
<div class="card">
    <h2>📝 Raw Data</h2>
    <details>
        <summary style="cursor: pointer; color: var(--accent-primary);">
            Click to expand JSON data
        </summary>
        <pre style="background: var(--bg-tertiary); padding: 1rem; border-radius: 0.5rem; 
                    overflow-x: auto; margin-top: 1rem; font-size: 0.8rem;">
{html.escape(json_data)}
        </pre>
    </details>
</div>
"""

    def _generate_footer(self) -> str:
        """Generate footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
</main>
<footer>
    Report generated on {timestamp} | RAG-Advanced Evaluation Framework
</footer>
"""


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_html_report(
    report: BenchmarkReport,
    title: str = "RAG Strategy Benchmark Report",
    config: HtmlReportConfig | None = None,
) -> str:
    """
    Generate an HTML report.

    Args:
        report: Benchmark report.
        title: Report title.
        config: Optional configuration.

    Returns:
        HTML string.
    """
    generator = HtmlReportGenerator(config)
    return generator.generate(report, title)


async def save_html_report(
    report: BenchmarkReport,
    path: Path | str,
    title: str = "RAG Strategy Benchmark Report",
    config: HtmlReportConfig | None = None,
) -> None:
    """
    Save an HTML report to file.

    Args:
        report: Benchmark report.
        path: Output file path.
        title: Report title.
        config: Optional configuration.
    """
    generator = HtmlReportGenerator(config)
    await generator.save(report, path, title)
