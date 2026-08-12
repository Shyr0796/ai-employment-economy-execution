#!/usr/bin/env python3
"""Generate publication-ready E0-E6 PDF figures within each cognitive regime.

The module reads the canonical monthly Parquet dataset, validates the
experiment matrix, and writes both individual vector figures and combined
atlases. Use ``--validate-only`` for a fast preflight with no output writes.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter, MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    ROOT
    / "output"
    / "research_data"
    / "institutional_v2_R1_R2_E0_E6_monthly.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "output" / "pdf" / "scientific_figures" / "institutional_v2"
)

REGIMES = {
    "R1": {
        "folder": "R1_government_llm",
        "title": "R1 - Government LLM",
        "definition": "Government LLM; rule-driven firms and residents",
    },
    "R2": {
        "folder": "R2_firm_government_llm",
        "title": "R2 - Firm and Government LLM",
        "definition": "Firm and government LLM; rule-driven residents",
    },
}

SCENARIOS = {
    "E0": "No new AI",
    "E1": "Laissez-faire AI",
    "E2": "Employment responsibility",
    "E3": "AI levy and social return",
    "E4": "Time dividend and solo firms",
    "E5": "Integrated social compact",
    "E6": "Compact under fiscal limits",
}

COLORS = {
    "E0": "#4D4D4D",
    "E1": "#0072B2",
    "E2": "#E69F00",
    "E3": "#009E73",
    "E4": "#CC79A7",
    "E5": "#D55E00",
    "E6": "#7A5195",
}

LINESTYLES = {
    "E0": (0, (5, 2)),
    "E1": "-",
    "E2": (0, (4, 2)),
    "E3": "-.",
    "E4": ":",
    "E5": "-",
    "E6": (0, (3, 1, 1, 1)),
}

ACTIVATION_MONTH = 25


@dataclass(frozen=True)
class MetricSpec:
    section_no: int
    section: str
    order: int
    slug: str
    column: str
    title: str
    ylabel: str
    unit_kind: str = "number"
    transform: str = "identity"
    note: str = ""


@dataclass(frozen=True)
class DatasetContext:
    """Provenance values displayed consistently on every figure."""

    source_name: str
    population: int
    seed: int
    first_month: int
    last_month: int


METRICS = [
    # 01 Production and effective demand
    MetricSpec(1, "production_demand", 1, "real_consumption", "real_consumption", "Real consumption", "Model currency", "money"),
    MetricSpec(1, "production_demand", 2, "firm_sales", "firm_sales", "Firm sales", "Model currency", "money"),
    MetricSpec(1, "production_demand", 3, "unmet_final_demand", "unmet_final_demand", "Unmet final demand", "Model currency", "money"),
    MetricSpec(1, "production_demand", 4, "capacity_utilization", "capacity_utilization", "Capacity utilization", "Percent", "percent"),
    MetricSpec(1, "production_demand", 5, "aggregate_price", "aggregate_price", "Aggregate price index", "Index", "index"),
    # 02 Household welfare and distribution
    MetricSpec(2, "household_distribution", 1, "disposable_income", "disposable_income", "Disposable income", "Model currency", "money"),
    MetricSpec(2, "household_distribution", 2, "bottom60_real_consumption", "bottom60_real_consumption", "Bottom-60% real consumption", "Model currency", "money"),
    MetricSpec(2, "household_distribution", 3, "bottom80_real_consumption", "bottom80_real_consumption", "Bottom-80% real consumption", "Model currency", "money"),
    MetricSpec(2, "household_distribution", 4, "liquidity_vulnerable_rate", "liquidity_vulnerable_rate", "Liquidity vulnerability", "Percent of households", "percent"),
    MetricSpec(2, "household_distribution", 5, "essential_cash_shortfall_rate", "essential_cash_shortfall_rate", "Essential-cash shortfall", "Percent of households", "percent"),
    MetricSpec(2, "household_distribution", 6, "economic_stress_rate", "economic_stress_rate", "Economic stress", "Percent of households", "percent"),
    MetricSpec(2, "household_distribution", 7, "disposable_income_atkinson", "disposable_income_atkinson_1_0", "Disposable-income inequality", "Atkinson index (epsilon = 1)", "index"),
    MetricSpec(2, "household_distribution", 8, "real_consumption_atkinson", "real_consumption_atkinson_1_0", "Real-consumption inequality", "Atkinson index (epsilon = 1)", "index"),
    # 03 Employment and work
    MetricSpec(3, "employment_work", 1, "unemployment_rate", "unemployment_rate", "Unemployment rate", "Percent", "percent"),
    MetricSpec(3, "employment_work", 2, "employment_rate", "employment_rate", "Employment rate", "Percent", "percent"),
    MetricSpec(3, "employment_work", 3, "wage_employment_rate", "wage_employment_rate", "Wage-employment rate", "Percent", "percent"),
    MetricSpec(3, "employment_work", 4, "self_employment_rate", "self_employment_rate", "Self-employment rate", "Percent", "percent"),
    MetricSpec(3, "employment_work", 5, "required_work_hours", "average_required_work_hours", "Required work hours", "Hours per month", "number"),
    MetricSpec(3, "employment_work", 6, "work_intensity", "average_work_intensity", "Average work intensity", "Index", "index"),
    MetricSpec(3, "employment_work", 7, "blocked_ai_layoffs_cumulative", "cumulative_ai_attributable_layoffs_blocked", "AI-attributable layoffs blocked", "Cumulative workers", "count"),
    MetricSpec(3, "employment_work", 8, "firm_exit_layoffs_cumulative", "firm_exit_layoffs", "Job losses from firm exits", "Cumulative workers", "count", "cumsum"),
    # 04 Firms and market structure
    MetricSpec(4, "firms_market", 1, "firm_count", "firm_count", "Active firm count", "Firms", "count"),
    MetricSpec(4, "firms_market", 2, "firm_entries_cumulative", "cumulative_firm_entries", "Firm entries", "Cumulative firms", "count"),
    MetricSpec(4, "firms_market", 3, "firm_exits_cumulative", "cumulative_firm_exits", "Firm exits", "Cumulative firms", "count"),
    MetricSpec(4, "firms_market", 4, "distressed_firm_share", "distressed_firm_share", "Distressed-firm share", "Percent of firms", "percent"),
    MetricSpec(4, "firms_market", 5, "firm_cash", "firm_cash", "Firm cash", "Model currency", "money"),
    MetricSpec(4, "firms_market", 6, "firm_bank_debt", "firm_bank_debt", "Firm bank debt", "Model currency", "money"),
    MetricSpec(4, "firms_market", 7, "market_hhi", "market_hhi", "Market concentration", "HHI", "index"),
    MetricSpec(4, "firms_market", 8, "firm_price_dispersion", "firm_price_dispersion", "Firm price dispersion", "Dispersion index", "index"),
    MetricSpec(4, "firms_market", 9, "below_cost_pricing_share", "below_cost_pricing_market_share", "Below-cost pricing market share", "Percent of market", "percent"),
    # 05 Solo enterprise and new demand
    MetricSpec(5, "solo_enterprise", 1, "solo_entries_cumulative", "cumulative_solo_entries", "Solo-enterprise entries", "Cumulative entries", "count"),
    MetricSpec(5, "solo_enterprise", 2, "solo_exits_cumulative", "cumulative_solo_exits", "Solo-enterprise exits", "Cumulative exits", "count"),
    MetricSpec(5, "solo_enterprise", 3, "solo_sales_cumulative", "solo_enterprise_sales", "Solo-enterprise sales", "Cumulative model currency", "money", "cumsum"),
    MetricSpec(5, "solo_enterprise", 4, "solo_income_cumulative", "solo_enterprise_income", "Solo-enterprise income", "Cumulative model currency", "money", "cumsum"),
    MetricSpec(5, "solo_enterprise", 5, "solo_net_new_demand_cumulative", "solo_net_additional_demand", "Solo-enterprise net new demand", "Cumulative model currency", "money", "cumsum"),
    MetricSpec(5, "solo_enterprise", 6, "solo_displacement_cumulative", "solo_incumbent_displacement", "Solo-enterprise incumbent displacement", "Cumulative model currency", "money", "cumsum"),
    # 06 Public service and fiscal system
    MetricSpec(6, "public_fiscal", 1, "public_service_index", "public_service_index", "Public-service index", "Index", "index"),
    MetricSpec(6, "public_fiscal", 2, "ai_levy_revenue_cumulative", "cumulative_ai_levy_revenue", "AI levy revenue", "Cumulative model currency", "money"),
    MetricSpec(6, "public_fiscal", 3, "ai_levy_public_service_cumulative", "cumulative_ai_levy_public_service_spending", "AI levy allocation to public service", "Cumulative model currency", "money"),
    MetricSpec(6, "public_fiscal", 4, "ai_levy_public_investment_cumulative", "cumulative_ai_levy_public_investment", "AI levy allocation to public investment", "Cumulative model currency", "money"),
    MetricSpec(6, "public_fiscal", 5, "government_fiscal_balance", "government_fiscal_balance", "Government fiscal balance", "Model currency per month", "money"),
    MetricSpec(6, "public_fiscal", 6, "government_debt_ratio", "government_debt_ratio", "Government debt ratio", "Percent", "percent"),
    MetricSpec(6, "public_fiscal", 7, "fiscal_curtailment_cumulative", "government_fiscal_curtailment", "Fiscal expenditure curtailment", "Cumulative model currency", "money", "cumsum"),
    MetricSpec(6, "public_fiscal", 8, "government_arrears_ratio", "government_arrears_ratio", "Government arrears ratio", "Percent", "percent"),
    # 07 Financial stability
    MetricSpec(7, "financial_stability", 1, "bank_credit_to_firms", "bank_credit_to_firms", "Bank credit to firms", "Model currency per month", "money"),
    MetricSpec(7, "financial_stability", 2, "bank_credit_rejected", "bank_firm_credit_rejected", "Rejected firm credit", "Model currency per month", "money"),
    MetricSpec(
        7,
        "financial_stability",
        3,
        "bank_capital_adequacy",
        "bank_capital_adequacy_ratio",
        "Bank capital adequacy",
        "Percent",
        "percent",
        "mask_undefined_999",
        "Values of +/-999 denote an undefined ratio when risk-weighted assets are zero and are not plotted.",
    ),
    MetricSpec(7, "financial_stability", 4, "bank_npl_ratio", "bank_npl_ratio", "Bank non-performing loan ratio", "Percent", "percent"),
    MetricSpec(7, "financial_stability", 5, "bank_reserve_ratio", "bank_reserve_ratio", "Bank reserve ratio", "Percent", "percent"),
    # 08 AI use and price behavior
    MetricSpec(8, "ai_price_behavior", 1, "personal_ai_use", "personal_ai_mean_use_rate", "Personal AI use", "Mean use rate (percent)", "percent"),
    MetricSpec(8, "ai_price_behavior", 2, "government_ai_use", "government_ai_use_rate", "Government AI use", "Use rate (percent)", "percent"),
    MetricSpec(8, "ai_price_behavior", 3, "firm_average_price", "firm_average_price", "Average firm price", "Price", "index"),
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "standard",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def transformed_values(frame: pd.DataFrame, spec: MetricSpec) -> pd.Series:
    values = pd.to_numeric(frame[spec.column], errors="coerce")
    if spec.transform == "cumsum":
        values = values.cumsum()
    elif spec.transform == "mask_undefined_999":
        values = values.mask(values.abs() >= 999.0)
    if spec.unit_kind == "percent":
        values = values * 100.0
    return values


def apply_axis_format(ax: plt.Axes, spec: MetricSpec) -> None:
    if spec.unit_kind == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    elif spec.unit_kind == "money":
        ax.yaxis.set_major_formatter(EngFormatter(places=2))
    elif spec.unit_kind == "count":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def full_y_axis_label(spec: MetricSpec) -> str:
    """Return a publication-safe y-axis label with metric identity and unit."""

    unit_labels = {
        "Percent": "%",
        "Percent of households": "% of households",
        "Percent of firms": "% of firms",
        "Percent of market": "% of market",
        "Mean use rate (percent)": "mean use rate, %",
        "Use rate (percent)": "use rate, %",
    }
    unit = unit_labels.get(spec.ylabel, spec.ylabel)
    title_lower = spec.title.lower()
    redundant_units = (
        (unit == "Index" and "index" in title_lower)
        or (unit == "Firms" and "firm count" in title_lower)
        or (unit == "Price" and "price" in title_lower)
    )
    if not unit or redundant_units:
        return spec.title
    if unit not in {"%", "HHI"}:
        unit = unit[0].lower() + unit[1:]
    return f"{spec.title}\n({unit})"


def pre_equilibrium_label_y(
    ax: plt.Axes,
    regime_data: pd.DataFrame,
    spec: MetricSpec,
) -> float:
    """Choose a two-line annotation height away from pre-equilibrium traces."""

    lower, upper = ax.get_ylim()
    span = upper - lower
    if not math.isfinite(span) or span <= 0:
        return 0.18

    normalized: list[float] = []
    for scenario in SCENARIOS:
        frame = regime_data[
            (regime_data["scenario_code"] == scenario)
            & (regime_data["month"] < ACTIVATION_MONTH)
        ].sort_values("month")
        values = transformed_values(frame, spec).to_numpy(dtype=float)
        normalized.extend(
            ((values[np.isfinite(values)] - lower) / span).clip(0.0, 1.0).tolist()
        )
    if not normalized:
        return 0.18

    candidates = (0.16, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88)
    return max(
        candidates,
        key=lambda candidate: (
            min(abs(candidate - value) for value in normalized),
            candidate,
        ),
    )


def add_standard_footer(
    fig: plt.Figure,
    regime_code: str,
    context: DatasetContext,
    extra: str = "",
) -> None:
    note = (
        f"Source: institutional_v2 formal; N={context.population}; "
        f"seed={context.seed}; months {context.first_month}-{context.last_month}. "
        f"{regime_code} E0-E6 within-architecture comparison; single-seed descriptive."
    )
    fig.text(
        0.08,
        0.038 if extra else 0.027,
        note,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    if extra:
        fig.text(
            0.08,
            0.021,
            extra,
            ha="left",
            va="bottom",
            fontsize=6.3,
            color="#666666",
        )


def plot_metric(
    data: pd.DataFrame,
    regime_code: str,
    spec: MetricSpec,
    context: DatasetContext,
    tex_ready: bool = False,
) -> plt.Figure:
    regime_data = data[data["regime_code"] == regime_code]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvspan(1, ACTIVATION_MONTH - 1, color="#F2F2F2", zorder=0)
    ax.axvline(
        ACTIVATION_MONTH,
        color="#666666",
        linewidth=0.9,
        linestyle=(0, (2, 2)),
        zorder=1,
    )
    for scenario in SCENARIOS:
        frame = regime_data[regime_data["scenario_code"] == scenario].sort_values("month")
        values = transformed_values(frame, spec)
        linewidth = 1.9 if scenario in {"E5", "E6"} else 1.55
        zorder = 4 if scenario in {"E5", "E6"} else 3
        ax.plot(
            frame["month"],
            values,
            color=COLORS[scenario],
            linestyle=LINESTYLES[scenario],
            linewidth=linewidth,
            label=scenario,
            zorder=zorder,
        )
    ax.set_xlim(1, 120)
    ax.set_xticks([1, 24, 48, 72, 96, 120])
    ax.set_xlabel("Simulation month")
    ax.set_ylabel(full_y_axis_label(spec), labelpad=9)
    if not tex_ready:
        ax.set_title(
            f"{REGIMES[regime_code]['title']} | {spec.title}",
            loc="left",
            pad=37,
        )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015, 1.0, 0.16),
        mode="expand",
        ncol=7,
        borderaxespad=0,
        handlelength=2.5,
        columnspacing=1.0,
    )
    ax.text(
        ACTIVATION_MONTH + 1.2,
        0.98,
        "AI / policy activation",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7,
        color="#555555",
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.94,
        },
        zorder=8,
    )
    apply_axis_format(ax, spec)
    if spec.unit_kind in {"percent", "count"}:
        lower, upper = ax.get_ylim()
        if lower > 0:
            ax.set_ylim(bottom=0)
        elif math.isclose(lower, upper):
            ax.set_ylim(lower - 1, upper + 1)
    ax.text(
        12.5,
        pre_equilibrium_label_y(ax, regime_data, spec),
        "Common\npre-equilibrium",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="center",
        fontsize=7.0,
        linespacing=1.0,
        color="#666666",
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": "#F2F2F2",
            "edgecolor": "none",
            "alpha": 0.94,
        },
        zorder=8,
    )
    extra = ""
    if spec.transform == "cumsum" or "Cumulative" in spec.ylabel:
        extra = "Cumulative series are explicitly identified in the title and y-axis."
    if spec.note:
        extra += " " + spec.note
    if not tex_ready:
        add_standard_footer(fig, regime_code, context, extra)
        fig.tight_layout(rect=(0.06, 0.09, 0.96, 0.9))
    else:
        fig.tight_layout(rect=(0.045, 0.045, 0.985, 0.91))
    return fig


def post_shock_summary(data: pd.DataFrame, regime_code: str) -> dict[str, list[float]]:
    subset = data[
        (data["regime_code"] == regime_code) & (data["month"] >= ACTIVATION_MONTH)
    ]
    result = {
        "bottom60_cumulative": [],
        "peak_unemployment": [],
        "tail_liquidity": [],
        "tail_public_service": [],
        "ending_firms": [],
        "max_debt_ratio": [],
    }
    for scenario in SCENARIOS:
        frame = subset[subset["scenario_code"] == scenario].sort_values("month")
        result["bottom60_cumulative"].append(frame["bottom60_real_consumption"].sum() / 1e6)
        result["peak_unemployment"].append(frame["unemployment_rate"].max() * 100)
        result["tail_liquidity"].append(frame.tail(12)["liquidity_vulnerable_rate"].mean() * 100)
        result["tail_public_service"].append(frame.tail(12)["public_service_index"].mean())
        result["ending_firms"].append(float(frame.iloc[-1]["firm_count"]))
        result["max_debt_ratio"].append(frame["government_debt_ratio"].max() * 100)
    return result


def annotate_bars(ax: plt.Axes, bars, fmt: Callable[[float], str]) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            fmt(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
        )


def plot_core_summary(
    data: pd.DataFrame,
    regime_code: str,
    context: DatasetContext,
    tex_ready: bool = False,
) -> plt.Figure:
    values = post_shock_summary(data, regime_code)
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.4))
    panels = [
        ("Bottom-60% cumulative real consumption", "Million model currency", "bottom60_cumulative", lambda value: f"{value:.1f}"),
        ("Peak unemployment", "Percent", "peak_unemployment", lambda value: f"{value:.1f}%"),
        ("Tail liquidity vulnerability", "Percent", "tail_liquidity", lambda value: f"{value:.1f}%"),
        ("Tail public-service index", "Index", "tail_public_service", lambda value: f"{value:.2f}"),
        ("Ending firm count", "Firms", "ending_firms", lambda value: f"{value:.0f}"),
        ("Maximum government debt ratio", "Percent", "max_debt_ratio", lambda value: f"{value:.2f}%"),
    ]
    x = np.arange(len(SCENARIOS))
    scenario_codes = list(SCENARIOS)
    for ax, (title, ylabel, key, formatter) in zip(axes.flat, panels):
        bars = ax.bar(
            x,
            values[key],
            color=[COLORS[scenario] for scenario in scenario_codes],
            edgecolor="white",
            linewidth=0.6,
            width=0.72,
        )
        ax.set_title(
            textwrap.fill(title, width=28),
            loc="left",
            fontsize=8.6,
            pad=7,
            linespacing=1.1,
        )
        ax.set_ylabel(ylabel, fontsize=7.8)
        ax.set_xticks(x, scenario_codes)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylim(bottom=0)
        annotate_bars(ax, bars, formatter)
    if tex_ready:
        fig.tight_layout(
            rect=(0.04, 0.045, 0.985, 0.985),
            h_pad=2.0,
            w_pad=1.8,
        )
    else:
        fig.suptitle(
            f"{REGIMES[regime_code]['title']} | Core economic outcomes",
            x=0.06,
            y=0.985,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.06,
            0.945,
            "E0-E6 are compared within one cognitive architecture; no cross-architecture values are mixed.",
            ha="left",
            fontsize=8.2,
            color="#444444",
        )
        add_standard_footer(
            fig,
            regime_code,
            context,
            "Tail values are last-12-month means; peak and maximum values use months 25-120.",
        )
        fig.tight_layout(
            rect=(0.04, 0.09, 0.98, 0.91),
            h_pad=2.0,
            w_pad=1.8,
        )
    return fig


def plot_cover(
    regime_code: str,
    metric_count: int,
    context: DatasetContext,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(
        0.1,
        0.9,
        "INSTITUTIONAL_V2 | SCIENTIFIC FIGURE ATLAS",
        fontsize=10,
        color="#555555",
        weight="bold",
    )
    fig.text(
        0.1,
        0.83,
        f"{REGIMES[regime_code]['title']}\nE0-E6 Monthly Comparisons",
        fontsize=25,
        color="#1F1F1F",
        weight="bold",
        linespacing=1.25,
    )
    fig.text(
        0.1,
        0.75,
        REGIMES[regime_code]["definition"],
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.1,
        0.69,
        f"{metric_count} metric figures + 1 core-outcome summary",
        fontsize=11,
        color="#1F1F1F",
        weight="bold",
    )
    scenario_lines = [
        f"{code}  {name}" for code, name in SCENARIOS.items()
    ]
    fig.text(
        0.1,
        0.61,
        "Scenario legend",
        fontsize=11,
        color="#1F1F1F",
        weight="bold",
    )
    y = 0.575
    for code, line in zip(SCENARIOS, scenario_lines):
        fig.add_artist(
            mpl.lines.Line2D(
                [0.1, 0.15],
                [y, y],
                transform=fig.transFigure,
                color=COLORS[code],
                linestyle=LINESTYLES[code],
                linewidth=2,
            )
        )
        fig.text(0.17, y - 0.005, line, fontsize=9.5, color="#333333")
        y -= 0.035
    fig.text(
        0.1,
        0.28,
        "Design and interpretation",
        fontsize=11,
        color="#1F1F1F",
        weight="bold",
        va="top",
    )
    fig.text(
        0.1,
        0.225,
        f"Population: {context.population} households | "
        f"Months: {context.first_month}-{context.last_month} | "
        f"Seed: {context.seed}\n"
        "Common pre-equilibrium: months 1-24 | AI and policy activation: month 25\n"
        "All figures are single-seed descriptive evidence, not uncertainty-qualified policy rankings.",
        fontsize=9.2,
        color="#444444",
        linespacing=1.6,
        va="top",
    )
    fig.text(
        0.1,
        0.08,
        f"Source: {context.source_name}",
        fontsize=7.5,
        color="#666666",
    )
    plt.axis("off")
    return fig


def safe_filename(regime_code: str, spec: MetricSpec) -> str:
    return (
        f"{regime_code}_{spec.section_no:02d}_{spec.order:02d}_"
        f"{spec.slug}.pdf"
    )


def required_columns() -> set[str]:
    """Return the canonical columns required by this atlas."""

    return {
        "regime_code",
        "scenario_code",
        "seed",
        "population",
        "month",
        *(spec.column for spec in METRICS),
    }


def validate_dataset(
    data: pd.DataFrame,
    regime_codes: Sequence[str] = tuple(REGIMES),
) -> None:
    """Fail fast when the requested experiment matrix is incomplete."""

    if data.empty:
        raise ValueError("Canonical plotting dataset is empty.")
    unknown_regimes = sorted(set(regime_codes) - set(REGIMES))
    if unknown_regimes:
        raise ValueError(f"Unknown cognitive regimes: {unknown_regimes}")

    missing = sorted(required_columns() - set(data.columns))
    if missing:
        raise ValueError(f"Required plotting columns are missing: {missing}")

    seeds = sorted(data["seed"].dropna().unique().tolist())
    if seeds != [1]:
        raise ValueError(
            "The current atlas renders single-seed trajectories only; "
            f"expected seed [1], found {seeds}. Add uncertainty aggregation "
            "before using a multi-seed dataset."
        )

    requested = data[data["regime_code"].isin(regime_codes)]
    duplicate_keys = requested.duplicated(
        ["regime_code", "scenario_code", "seed", "month"]
    )
    if duplicate_keys.any():
        examples = requested.loc[
            duplicate_keys,
            ["regime_code", "scenario_code", "seed", "month"],
        ].head(5)
        raise ValueError(
            "Duplicate regime/scenario/seed/month rows found: "
            f"{examples.to_dict(orient='records')}"
        )

    expected_scenarios = set(SCENARIOS)
    reference_months: tuple[int, ...] | None = None
    for regime_code in regime_codes:
        regime_data = requested[requested["regime_code"] == regime_code]
        found_scenarios = set(regime_data["scenario_code"].unique())
        if found_scenarios != expected_scenarios:
            raise ValueError(
                f"{regime_code} scenario matrix mismatch; "
                f"missing={sorted(expected_scenarios - found_scenarios)}, "
                f"extra={sorted(found_scenarios - expected_scenarios)}"
            )
        for scenario_code in SCENARIOS:
            months = tuple(
                sorted(
                    regime_data.loc[
                        regime_data["scenario_code"] == scenario_code,
                        "month",
                    ]
                    .astype(int)
                    .tolist()
                )
            )
            if not months or months[0] != 1 or ACTIVATION_MONTH not in months:
                raise ValueError(
                    f"{regime_code}/{scenario_code} must contain month 1 and "
                    f"activation month {ACTIVATION_MONTH}."
                )
            if months != tuple(range(months[0], months[-1] + 1)):
                raise ValueError(
                    f"{regime_code}/{scenario_code} has a non-contiguous month series."
                )
            if reference_months is None:
                reference_months = months
            elif months != reference_months:
                raise ValueError(
                    f"{regime_code}/{scenario_code} does not share the common month grid."
                )


def load_dataset(
    data_path: Path,
    regime_codes: Sequence[str] = tuple(REGIMES),
) -> pd.DataFrame:
    """Load and validate the canonical monthly dataset."""

    if not data_path.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {data_path}")
    data = pd.read_parquet(data_path)
    validate_dataset(data, regime_codes)
    return data


def build_context(data: pd.DataFrame, data_path: Path) -> DatasetContext:
    """Derive the provenance displayed on every output page."""

    populations = sorted(data["population"].dropna().astype(int).unique().tolist())
    seeds = sorted(data["seed"].dropna().astype(int).unique().tolist())
    if len(populations) != 1 or len(seeds) != 1:
        raise ValueError(
            "Figure provenance requires exactly one population and one seed; "
            f"found populations={populations}, seeds={seeds}."
        )
    return DatasetContext(
        source_name=data_path.name,
        population=populations[0],
        seed=seeds[0],
        first_month=int(data["month"].min()),
        last_month=int(data["month"].max()),
    )


def relative_or_absolute(path: Path) -> str:
    """Prefer project-relative catalog paths while supporting custom outputs."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def generate(
    data_path: Path = DEFAULT_DATA_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    regime_codes: Sequence[str] = tuple(REGIMES),
    tex_ready: bool = False,
) -> list[dict[str, str]]:
    """Generate individual figures and combined atlases."""

    configure_style()
    data_path = data_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    regime_codes = tuple(regime_codes)
    data = load_dataset(data_path, regime_codes)
    context = build_context(data, data_path)

    output_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, str]] = []
    for regime_code in regime_codes:
        regime = REGIMES[regime_code]
        regime_dir = output_root / regime["folder"]
        regime_dir.mkdir(parents=True, exist_ok=True)
        atlas_path = output_root / f"{regime_code}_E0-E6_scientific_figure_atlas.pdf"
        with PdfPages(atlas_path) as atlas:
            metadata = atlas.infodict()
            metadata["Title"] = f"{regime['title']} E0-E6 Scientific Figure Atlas"
            metadata["Author"] = "Urban Cup research project"
            metadata["Subject"] = "Institutional_v2 monthly economic indicators"
            metadata["Keywords"] = "AI economy, agent-based model, E0-E6, R1, R2"

            if not tex_ready:
                cover = plot_cover(regime_code, len(METRICS), context)
                atlas.savefig(cover, bbox_inches=None)
                plt.close(cover)

            summary_dir = regime_dir / "00_core_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / f"{regime_code}_00_core_outcomes_summary.pdf"
            summary = plot_core_summary(data, regime_code, context, tex_ready)
            summary.savefig(
                summary_path,
                bbox_inches=None if tex_ready else "tight",
            )
            atlas.savefig(summary, bbox_inches=None)
            plt.close(summary)
            catalog.append(
                {
                    "regime": regime_code,
                    "section": "00_core_summary",
                    "number": "00-00",
                    "metric": "Six primary outcomes",
                    "transform": "post-shock aggregation",
                    "path": relative_or_absolute(summary_path),
                }
            )

            for spec in METRICS:
                section_dir = regime_dir / f"{spec.section_no:02d}_{spec.section}"
                section_dir.mkdir(parents=True, exist_ok=True)
                figure_path = section_dir / safe_filename(regime_code, spec)
                figure = plot_metric(
                    data,
                    regime_code,
                    spec,
                    context,
                    tex_ready,
                )
                figure.savefig(
                    figure_path,
                    bbox_inches=None if tex_ready else "tight",
                )
                atlas.savefig(figure, bbox_inches=None)
                plt.close(figure)
                catalog.append(
                    {
                        "regime": regime_code,
                        "section": f"{spec.section_no:02d}_{spec.section}",
                        "number": f"{spec.section_no:02d}-{spec.order:02d}",
                        "metric": spec.title,
                        "transform": spec.transform,
                        "path": relative_or_absolute(figure_path),
                    }
                )
        catalog.append(
            {
                "regime": regime_code,
                "section": "atlas",
                "number": "ALL",
                "metric": "Combined multi-page atlas",
                "transform": "none",
                "path": relative_or_absolute(atlas_path),
            }
        )
    return catalog


def write_index(
    catalog: list[dict[str, str]],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    data_path: Path = DEFAULT_DATA_PATH,
    regime_codes: Sequence[str] = tuple(REGIMES),
    tex_ready: bool = False,
) -> Path:
    """Write the figure catalog beside the generated PDFs."""

    output_root = output_root.expanduser().resolve()
    index_path = output_root / "FIGURE_INDEX.md"
    lines = [
        "# Institutional v2 Scientific PDF Figure Index",
        "",
        (
            "All figures use a white-background scientific style. R1 and R2 "
            "are stored separately; each metric figure compares E0-E6 within "
            "one cognitive architecture."
        ),
        "",
        "## Combined atlases",
        "",
        *[
            f"- `{regime_code}_E0-E6_scientific_figure_atlas.pdf`"
            for regime_code in regime_codes
        ],
        "",
        (
            "Each atlas contains "
            + (
                "one six-indicator core summary and "
                if tex_ready
                else "a cover, one six-indicator core summary, and "
            )
            + "one monthly E0-E6 comparison page per selected metric. "
            "Individual vector PDFs are organized below the regime folders."
        ),
        "",
        *(
            [
                "## TeX-ready presentation contract",
                "",
                "- No regime/metric title is embedded above an individual figure.",
                "- No source or provenance footer is embedded below a figure.",
                "- Every time-series y-axis states the metric name and unit.",
                "- The common pre-equilibrium annotation is two lines and collision-aware.",
                "",
            ]
            if tex_ready
            else []
        ),
        "",
        "## Naming contract",
        "",
        "`{regime}_{section}_{order}_{metric_slug}.pdf`",
        "",
        "Example: `R2_03_01_unemployment_rate.pdf`.",
        "",
        "## Figure catalog",
        "",
        "| Regime | Number | Section | Metric | Transformation | PDF |",
        "|---|---|---|---|---|---|",
    ]
    for item in catalog:
        lines.append(
            f"| {item['regime']} | {item['number']} | {item['section']} | "
            f"{item['metric']} | {item['transform']} | `{item['path']}` |"
        )
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- Months 1-24 are the common pre-equilibrium period; the vertical reference line marks month 25.",
        "- Rates and ratios are displayed as percentages; the Parquet source stores them as proportions.",
        "- Cumulative figures are either source cumulative fields or an explicitly documented cumulative sum.",
        "- E0-E6 comparisons are within R1 or within R2. No chart mixes cognitive architectures.",
        "- Current figures use seed 1 and must be treated as descriptive trajectories.",
        "",
        "## Regeneration",
        "",
        "```bash",
        ".venv/bin/python ai_economy_execution/reporting/generate_scientific_metric_atlas.py \\",
        f"  --data {relative_or_absolute(data_path.expanduser().resolve())} \\",
        f"  --output {relative_or_absolute(output_root)}"
        + (" \\" if tex_ready else ""),
        *(["  --tex-ready"] if tex_ready else []),
        "```",
        "",
    ]
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the stable command-line interface."""

    parser = argparse.ArgumentParser(
        description="Generate scientific E0-E6 PDF comparisons within R1 and/or R2."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Canonical monthly Parquet dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for atlases, individual PDFs, and index.",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(REGIMES),
        default=list(REGIMES),
        help="Cognitive regimes to render (default: R1 R2).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset and matrix without writing figures.",
    )
    parser.add_argument(
        "--tex-ready",
        action="store_true",
        help=(
            "Suppress embedded figure titles and source footers, use complete "
            "y-axis labels, and omit atlas covers for direct TeX insertion."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    regime_codes = tuple(args.regimes)
    if args.validate_only:
        data_path = args.data.expanduser().resolve()
        data = load_dataset(data_path, regime_codes)
        context = build_context(data, data_path)
        print(
            "Validation passed: "
            f"rows={len(data)}, columns={len(data.columns)}, "
            f"regimes={','.join(regime_codes)}, scenarios=E0-E6, "
            f"seed={context.seed}, months={context.first_month}-{context.last_month}"
        )
        return

    catalog = generate(
        args.data,
        args.output,
        regime_codes,
        tex_ready=args.tex_ready,
    )
    index_path = write_index(
        catalog,
        args.output,
        args.data,
        regime_codes,
        tex_ready=args.tex_ready,
    )
    individual = [item for item in catalog if item["number"] not in {"ALL"}]
    print(f"Output root: {args.output.expanduser().resolve()}")
    print(f"Individual PDFs: {len(individual)}")
    print(f"Combined atlases: {len(regime_codes)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
