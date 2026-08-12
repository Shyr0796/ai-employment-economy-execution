#!/usr/bin/env python3
"""Generate the paper data bundle and SVG figures for institutional_v2 R1/R2."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "ai_economy_execution"
    / "results"
    / "research_matrix"
    / "formal"
    / "institutional_v2"
)
RUN = Path("hkust_gpt-3-5-turbo") / "N00500_M120_S001"
FIGURES = ROOT / "output" / "paper_figures" / "institutional_v2"
SCENARIOS = [f"E{i}" for i in range(7)]
REGIMES = {
    "R1": "R1_government",
    "R2": "R2_firm_government",
}

BG = "#ffffff"
SURFACE = "#f7f9fb"
RAISED = "#eef3f8"
TEXT = "#1c2733"
SECONDARY = "#5f6d7a"
GRID = "#d8e0e8"
FOCUS = "#2868b7"
POSITIVE = "#167a63"
WARNING = "#9a6500"
NEGATIVE = "#b83b49"
NEUTRAL = "#8794a1"

SCENARIO_NAMES = {
    "E0": "No New AI",
    "E1": "Laissez-Faire AI",
    "E2": "Employment Responsibility",
    "E3": "AI Levy & Social Return",
    "E4": "Time Dividend & Solo Firms",
    "E5": "Integrated Social Compact",
    "E6": "Compact, Fiscal Constraint",
}

MECHANISMS = [
    "Private AI",
    "Employment\nresponsibility",
    "AI levy &\nsocial return",
    "Solo-enterprise\nchannel",
    "Active demand &\ngov. AI",
    "Tighter fiscal\nconstraint",
]

DESIGN = {
    "E0": [0, 0, 0, 0, 0, 0],
    "E1": [1, 0, 0, 0, 0, 0],
    "E2": [1, 1, 0, 0, 0, 0],
    "E3": [1, 0, 1, 0, 0, 0],
    "E4": [1, 0, 0, 1, 0, 0],
    "E5": [1, 1, 1, 1, 1, 0],
    "E6": [1, 1, 1, 1, 1, 1],
}

QUESTIONS = {
    "E0": "No-shock counterfactual",
    "E1": "What does private AI alone do?",
    "E2": "Can firms retain workers and share costs?",
    "E3": "Can AI rents finance public services?",
    "E4": "Can saved time become independent demand?",
    "E5": "Do the mechanisms work as a portfolio?",
    "E6": "Does the portfolio survive tighter fiscal rules?",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 14,
    fill: str = TEXT,
    weight: int = 400,
    anchor: str = "start",
    family: str = "Inter,Arial,'Noto Sans SC',sans-serif",
    opacity: float = 1.0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" '
        f'opacity="{opacity}">{esc(value)}</text>'
    )


def multiline(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    fill: str = SECONDARY,
    weight: int = 400,
    anchor: str = "start",
    leading: float = 1.25,
) -> str:
    lines = value.split("\n")
    spans = "".join(
        f'<tspan x="{x:.1f}" dy="{0 if i == 0 else size * leading:.1f}">{esc(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,\'Noto Sans SC\',sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{spans}</text>'
    )


def svg_start(width: int, height: int, title_value: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{esc(title_value)}</title>",
        f"<desc id=\"desc\">{esc(desc)}</desc>",
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
    ]


def save_svg(name: str, parts: Iterable[str]) -> None:
    path = FIGURES / name
    path.write_text("\n".join([*parts, "</svg>"]) + "\n", encoding="utf-8")


def load_runs() -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for regime, dirname in REGIMES.items():
        runs[regime] = {}
        for scenario in SCENARIOS:
            folder = RESULTS / dirname / RUN / scenario
            summary_doc = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            behavior_doc = json.loads(
                (folder / "decision_behavior_summary.json").read_text(encoding="utf-8")
            )
            with (folder / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                monthly = list(csv.DictReader(handle))
            runs[regime][scenario] = {
                "folder": str(folder.relative_to(ROOT)),
                "summary": summary_doc["summary"],
                "decision_audit": summary_doc["decision_audit"],
                "behavior": behavior_doc,
                "monthly": monthly,
                "source_fingerprint": summary_doc["source_fingerprint"],
                "provider": summary_doc["provider"],
            }
    return runs


def pct(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def pp(value: float, reference: float) -> float:
    return 100.0 * (value - reference)


def arrow(
    p: list[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = FOCUS,
    width: float = 3,
    dashed: bool = False,
) -> None:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    p.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"{dash}/>'
    )


def framework_card(
    p: list[str],
    x: float,
    y: float,
    width: float,
    title_value: str,
    body: str,
    *,
    accent: str = FOCUS,
) -> None:
    p.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="150" rx="14" '
        f'fill="{SURFACE}" stroke="{GRID}"/>'
    )
    p.append(f'<rect x="{x}" y="{y}" width="6" height="150" rx="3" fill="{accent}"/>')
    p.append(text(x + 22, y + 36, title_value, size=17, weight=750))
    p.append(multiline(x + 22, y + 70, body, size=12, fill=SECONDARY, leading=1.45))


def figure_framework() -> None:
    width, height = 1500, 920
    p = svg_start(
        width,
        height,
        "Figure 1. Closed-loop AI economy and institutional channels",
        "A closed monthly feedback loop connects AI adoption, firms, households, effective demand, "
        "business dynamics, government, and finance. E2 through E6 alter specific transmission channels.",
    )
    p.append(
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{FOCUS}"/></marker></defs>'
    )
    p += [
        text(64, 62, "FIGURE 1  |  MODEL FRAMEWORK", size=14, fill=FOCUS, weight=700),
        text(
            64,
            106,
            "AI productivity becomes shared welfare only through a closed demand loop",
            size=31,
            weight=700,
        ),
        text(
            64,
            138,
            "Institutions change where transition constraints fall across households, firms, and government.",
            size=16,
            fill=SECONDARY,
        ),
    ]

    y = 215
    framework_card(p, 64, y, 225, "AI Adoption", "Productivity\nCapacity\nPrice advantage", accent=FOCUS)
    framework_card(p, 330, y, 245, "Firms & Markets", "Prices and wages\nLabor demand\nProduction", accent=FOCUS)
    framework_card(p, 616, y, 245, "Households", "Employment and income\nCash and liquidity\nWork time", accent=WARNING)
    framework_card(p, 902, y, 245, "Effective Demand", "Consumption\nPublic procurement\nNew demand", accent=POSITIVE)
    framework_card(p, 1188, y, 248, "Business Dynamics", "Sales and profits\nInvestment\nEntry and exit", accent=NEGATIVE)

    for x1, x2 in ((289, 330), (575, 616), (861, 902), (1147, 1188)):
        arrow(p, x1, y + 75, x2, y + 75)
    p += [
        text(309, y + 62, "capacity", size=10, fill=SECONDARY, anchor="middle"),
        text(595, y + 62, "jobs / wages", size=10, fill=SECONDARY, anchor="middle"),
        text(881, y + 62, "spending", size=10, fill=SECONDARY, anchor="middle"),
        text(1167, y + 62, "orders", size=10, fill=SECONDARY, anchor="middle"),
        f'<path d="M 1312 {y + 150} C 1312 {y + 205}, 480 {y + 205}, 455 {y + 152}" '
        f'fill="none" stroke="{FOCUS}" stroke-width="3" marker-end="url(#arrow)"/>',
        text(
            905,
            y + 250,
            "sales, profits, investment, entry and exit feed back into firm decisions",
            size=11,
            fill=SECONDARY,
            anchor="middle",
        ),
    ]

    def tag(x: float, y_tag: float, label: str, color: str) -> None:
        tag_w = max(92, 8.2 * len(label) + 22)
        p.append(
            f'<rect x="{x}" y="{y_tag}" width="{tag_w}" height="28" rx="14" '
            f'fill="{color}" opacity=".18" stroke="{color}"/>'
        )
        p.append(text(x + tag_w / 2, y_tag + 19, label, size=10, fill=color, weight=750, anchor="middle"))

    tag(505, 176, "E2 employment responsibility", WARNING)
    tag(790, 176, "E4 time dividend / solo enterprise", POSITIVE)

    gov_x, gov_y, gov_w = 330, 500, 680
    p.append(
        f'<rect x="{gov_x}" y="{gov_y}" width="{gov_w}" height="155" rx="16" '
        f'fill="{SURFACE}" stroke="{GRID}"/>'
    )
    p += [
        text(gov_x + 24, gov_y + 36, "Government & Social Return", size=19, weight=750),
        text(gov_x + 24, gov_y + 68, "Taxes and AI levy", size=12, fill=SECONDARY),
        text(gov_x + 215, gov_y + 68, "Transfers and procurement", size=12, fill=SECONDARY),
        text(gov_x + 452, gov_y + 68, "Public services and investment", size=12, fill=SECONDARY),
        text(gov_x + 24, gov_y + 113, "E3", size=15, fill=POSITIVE, weight=800),
        text(gov_x + 62, gov_y + 113, "AI levy and earmarked return", size=13),
        text(gov_x + 315, gov_y + 113, "E5", size=15, fill=FOCUS, weight=800),
        text(gov_x + 353, gov_y + 113, "integrated social compact", size=13),
        text(gov_x + 535, gov_y + 113, "E6", size=15, fill=WARNING, weight=800),
        text(gov_x + 573, gov_y + 113, "fiscal limits", size=13),
    ]
    arrow(p, 455, y + 150, 455, gov_y, color=FOCUS, dashed=True)
    arrow(p, 730, gov_y, 730, y + 150, color=FOCUS, dashed=True)
    arrow(p, 895, gov_y, 1025, y + 150, color=FOCUS, dashed=True)
    p += [
        text(463, 444, "taxes / levy", size=10, fill=SECONDARY),
        text(738, 444, "transfers / services", size=10, fill=SECONDARY),
        text(944, 444, "procurement", size=10, fill=SECONDARY),
    ]

    fin_x, fin_y, fin_w = 1050, 500, 386
    p.append(
        f'<rect x="{fin_x}" y="{fin_y}" width="{fin_w}" height="155" rx="16" '
        f'fill="{SURFACE}" stroke="{GRID}"/>'
    )
    p += [
        text(fin_x + 24, fin_y + 36, "Bank & Finance", size=19, weight=750),
        text(fin_x + fin_w - 24, fin_y + 36, "cross-cutting", size=11, fill=FOCUS, weight=700, anchor="end"),
        multiline(
            fin_x + 24,
            fin_y + 72,
            "Firm credit and debt service\nGovernment borrowing capacity\nHousehold financial buffers",
            size=12,
            fill=SECONDARY,
            leading=1.5,
        ),
    ]

    p.append(text(64, 716, "EVALUATION LAYER", size=12, fill=SECONDARY, weight=750))
    evaluation = [
        ("Household Welfare", "consumption / employment / liquidity"),
        ("Firm & Market", "sales / entry-exit / concentration"),
        ("Distribution", "bottom 60% / Atkinson"),
        ("Fiscal & Financial", "public service / debt / curtailment"),
        ("Validity & Robustness", "audit / seeds / architecture"),
    ]
    eval_y, gap, card_w = 738, 14, 263
    for i, (label, body) in enumerate(evaluation):
        xx = 64 + i * (card_w + gap)
        p.append(
            f'<rect x="{xx}" y="{eval_y}" width="{card_w}" height="78" rx="10" '
            f'fill="{RAISED}" stroke="{GRID}"/>'
        )
        p.append(text(xx + 14, eval_y + 29, label, size=13, weight=700))
        p.append(text(xx + 14, eval_y + 54, body, size=10, fill=SECONDARY))
    p += [
        f'<line x1="64" y1="853" x2="1436" y2="853" stroke="{GRID}"/>',
        text(
            64,
            887,
            "Core interpretation: policy does not remove constraints; it redistributes them across jobs, demand, firms, and fiscal space.",
            size=14,
            weight=650,
        ),
    ]
    save_svg("figure_1_model_framework.svg", p)


def figure_design_matrix() -> None:
    width, height = 1500, 820
    p = svg_start(
        width,
        height,
        "Supplementary Figure S1. E0-E6 institutional design matrix",
        "A seven-by-six mechanism matrix defining the counterfactual meaning of scenarios E0 through E6.",
    )
    p += [
        text(64, 64, "SUPPLEMENTARY FIGURE S1  |  INSTITUTIONAL DESIGN", size=14, fill=FOCUS, weight=700),
        text(64, 108, "E0-E6 are policy mechanisms, not arbitrary labels", size=32, weight=700),
        text(
            64,
            139,
            "Each comparison answers a distinct causal question; downstream analysis must preserve these baselines.",
            size=16,
            fill=SECONDARY,
        ),
    ]
    x0, y0 = 64, 205
    scenario_w, cell_w, question_x = 260, 126, 1100
    p.append(f'<rect x="{x0}" y="{y0}" width="1372" height="540" rx="12" fill="{SURFACE}"/>')
    p.append(text(x0 + 18, y0 + 34, "SCENARIO / POLICY PACKAGE", size=12, fill=SECONDARY, weight=700))
    for col, label in enumerate(MECHANISMS):
        p.append(
            multiline(
                x0 + scenario_w + col * cell_w + cell_w / 2,
                y0 + 25,
                label,
                size=11,
                fill=SECONDARY,
                weight=700,
                anchor="middle",
            )
        )
    p.append(text(question_x, y0 + 34, "IDENTIFICATION QUESTION", size=12, fill=SECONDARY, weight=700))
    for row, scenario in enumerate(SCENARIOS):
        y = y0 + 70 + row * 65
        if scenario in {"E3", "E5", "E6"}:
            p.append(
                f'<rect x="{x0 + 8}" y="{y - 37}" width="1356" height="55" rx="8" '
                f'fill="{RAISED}" stroke="{FOCUS if scenario == "E3" else GRID}" stroke-width="1"/>'
            )
        p.append(text(x0 + 18, y, scenario, size=20, fill=FOCUS, weight=800))
        p.append(text(x0 + 64, y, SCENARIO_NAMES[scenario], size=14, weight=650))
        for col, active in enumerate(DESIGN[scenario]):
            cx = x0 + scenario_w + col * cell_w + cell_w / 2
            if active:
                fill = WARNING if scenario == "E6" and col == 5 else FOCUS
                p.append(f'<circle cx="{cx}" cy="{y - 6}" r="13" fill="{fill}" opacity=".95"/>')
                p.append(text(cx, y - 1, "ON", size=9, fill=BG, weight=900, anchor="middle"))
            else:
                p.append(f'<circle cx="{cx}" cy="{y - 6}" r="4" fill="{GRID}"/>')
        p.append(text(question_x, y, QUESTIONS[scenario], size=12, fill=SECONDARY))
    p += [
        f'<line x1="64" y1="772" x2="1436" y2="772" stroke="{GRID}"/>',
        text(64, 801, "Required contrasts:", size=12, fill=SECONDARY, weight=700),
        text(
            185,
            801,
            "E1-E0 = private-AI shock   |   E2/E3/E4-E1 = single mechanisms   |   E5 = integrated package   |   E6-E5 = fiscal constraint",
            size=12,
            fill=TEXT,
        ),
    ]
    save_svg("supplementary_figure_s1_design_matrix.svg", p)


def bar_panel(
    p: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title_value: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    unit: str,
    *,
    higher_is_better: bool,
) -> None:
    p.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{SURFACE}"/>')
    p.append(text(x + 18, y + 30, title_value, size=15, weight=700))
    p.append(text(x + 18, y + 51, subtitle, size=11, fill=SECONDARY))
    chart_x, chart_y = x + 52, y + 76
    chart_w, chart_h = width - 70, height - 112
    low = min(values + [0])
    high = max(values + [0])
    span = high - low or 1
    low -= span * 0.08
    high += span * 0.12
    zero_y = chart_y + (high / (high - low)) * chart_h
    p.append(f'<line x1="{chart_x}" y1="{zero_y:.1f}" x2="{chart_x + chart_w}" y2="{zero_y:.1f}" stroke="{GRID}"/>')
    step = chart_w / len(values)
    for i, value in enumerate(values):
        bx = chart_x + i * step + step * 0.18
        bw = step * 0.64
        vy = chart_y + (high - value) / (high - low) * chart_h
        top = min(vy, zero_y)
        bh = max(2.5, abs(zero_y - vy))
        scenario = labels[i]
        if abs(value) < 0.05:
            color = NEUTRAL
        else:
            improved = value > 0 if higher_is_better else value < 0
            color = POSITIVE if improved else NEGATIVE
        stroke = f' stroke="{FOCUS}" stroke-width="3"' if scenario == "E3" else ""
        p.append(
            f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'rx="3" fill="{color}"{stroke}/>'
        )
        p.append(text(bx + bw / 2, y + height - 18, scenario, size=11, fill=SECONDARY, anchor="middle"))
        label_y = top - 7 if value >= 0 else top + bh + 14
        formatted = f"{value:+.1f}{unit}" if unit else f"{value:.1f}"
        p.append(text(bx + bw / 2, label_y, formatted, size=10, fill=TEXT, weight=650, anchor="middle"))


def figure_r2_outcomes(runs: dict[str, dict[str, dict]]) -> None:
    r2 = runs["R2"]
    e0 = r2["E0"]["summary"]
    e1 = r2["E1"]["summary"]
    policies = ["E2", "E3", "E4", "E5", "E6"]
    metrics = {
        "consumption": [
            pct(r2[e]["summary"]["cumulative_real_consumption"], e1["cumulative_real_consumption"])
            for e in policies
        ],
        "bottom60": [
            pct(
                r2[e]["summary"]["bottom60_cumulative_real_consumption"],
                e1["bottom60_cumulative_real_consumption"],
            )
            for e in policies
        ],
        "peak_u": [
            pp(r2[e]["summary"]["peak_unemployment_rate"], e1["peak_unemployment_rate"])
            for e in policies
        ],
        "public_service": [
            pct(r2[e]["summary"]["tail_public_service_index"], e1["tail_public_service_index"])
            for e in policies
        ],
        "liquidity": [
            pp(
                r2[e]["summary"]["tail_liquidity_vulnerable_rate"],
                e1["tail_liquidity_vulnerable_rate"],
            )
            for e in policies
        ],
        "firms": [
            r2[e]["summary"]["ending_firm_count"] - e1["ending_firm_count"]
            for e in policies
        ],
    }

    shock_consumption = pct(
        e1["cumulative_real_consumption"], e0["cumulative_real_consumption"]
    )
    shock_bottom60 = pct(
        e1["bottom60_cumulative_real_consumption"],
        e0["bottom60_cumulative_real_consumption"],
    )
    shock_unemployment = pp(
        e1["peak_unemployment_rate"], e0["peak_unemployment_rate"]
    )

    width, height = 1500, 820
    p = svg_start(
        width,
        height,
        "Figure 2. R2 main policy outcomes",
        "A concise light-background evidence matrix compares the private-AI shock E1-E0 "
        "and policy scenarios E2 through E6 against laissez-faire AI E1.",
    )
    p += [
        text(64, 62, "FIGURE 2  |  R2 MAIN RESULTS", size=14, fill=FOCUS, weight=700),
        text(64, 106, "AI gains and transition costs coexist", size=31, weight=700),
        text(
            64,
            137,
            "Policy mechanisms redistribute welfare, employment, market, and fiscal pressures rather than improving every outcome.",
            size=16,
            fill=SECONDARY,
        ),
    ]
    p.append(f'<rect x="64" y="166" width="1372" height="94" rx="10" fill="{RAISED}" stroke="{GRID}"/>')
    p += [
        text(84, 192, "PRIVATE AI SHOCK  |  E1 VS E0", size=11, fill=FOCUS, weight=750),
        text(84, 230, f"{shock_consumption:+.1f}%", size=25, fill=POSITIVE, weight=800),
        text(196, 230, "real consumption", size=11, fill=SECONDARY),
        text(405, 230, f"{shock_bottom60:+.1f}%", size=25, fill=POSITIVE, weight=800),
        text(517, 230, "bottom-60% consumption", size=11, fill=SECONDARY),
        text(783, 230, f"{shock_unemployment:+.1f} pp", size=25, fill=NEGATIVE, weight=800),
        text(930, 230, "peak unemployment", size=11, fill=SECONDARY),
        text(1162, 214, "Growth and transition stress", size=12, weight=700),
        text(1162, 235, "occur on the same path.", size=12, fill=SECONDARY),
    ]

    grid_x, grid_y, grid_w = 64, 292, 1372
    header_h, row_h = 62, 68
    columns = [
        ("Scenario", 110),
        ("Real\nconsumption", 170),
        ("Bottom-60%\nconsumption", 170),
        ("Peak\nunemployment", 170),
        ("Public-service\nindex", 180),
        ("Liquidity\nvulnerability", 190),
        ("Ending\nfirms", 150),
        ("Main readout", 232),
    ]
    p.append(
        f'<rect x="{grid_x}" y="{grid_y}" width="{grid_w}" height="{header_h + row_h * len(policies)}" '
        f'rx="8" fill="{BG}" stroke="{GRID}"/>'
    )
    p.append(f'<rect x="{grid_x}" y="{grid_y}" width="{grid_w}" height="{header_h}" rx="8" fill="{SURFACE}"/>')
    cursor = grid_x
    for label, col_w in columns:
        p.append(multiline(cursor + 12, grid_y + 24, label, size=11, fill=SECONDARY, weight=700, leading=1.2))
        cursor += col_w
        p.append(
            f'<line x1="{cursor}" y1="{grid_y}" x2="{cursor}" '
            f'y2="{grid_y + header_h + row_h * len(policies)}" stroke="{GRID}"/>'
        )

    formats = [
        ("consumption", "%", True),
        ("bottom60", "%", True),
        ("peak_u", " pp", False),
        ("public_service", "%", True),
        ("liquidity", " pp", False),
        ("firms", "", True),
    ]
    readouts = {
        "E2": "Retention works;\nexit risk rises",
        "E3": "Most balanced\nin this path",
        "E4": "Liquidity improves;\ndemand shifts",
        "E5": "Services rise;\nfirms exit",
        "E6": "Debt falls vs E5;\nfirms do not recover",
    }
    good_fill, bad_fill, flat_fill, focus_fill = "#eaf6f1", "#fbecee", "#f3f5f7", "#eaf1fb"
    metric_widths = [columns[i][1] for i in range(1, 7)]
    for row, scenario in enumerate(policies):
        yy = grid_y + header_h + row * row_h
        row_fill = focus_fill if scenario == "E3" else BG
        p.append(f'<rect x="{grid_x}" y="{yy}" width="{grid_w}" height="{row_h}" fill="{row_fill}"/>')
        p.append(f'<line x1="{grid_x}" y1="{yy}" x2="{grid_x + grid_w}" y2="{yy}" stroke="{GRID}"/>')
        p.append(text(grid_x + 18, yy + 41, scenario, size=17, fill=FOCUS, weight=800))
        x_cell = grid_x + columns[0][1]
        for metric_index, (key, unit, higher_is_better) in enumerate(formats):
            value = metrics[key][row]
            col_w = metric_widths[metric_index]
            if abs(value) < 0.05:
                fill, value_color = flat_fill, NEUTRAL
            else:
                improved = value > 0 if higher_is_better else value < 0
                fill, value_color = (good_fill, POSITIVE) if improved else (bad_fill, NEGATIVE)
            p.append(
                f'<rect x="{x_cell + 7}" y="{yy + 9}" width="{col_w - 14}" height="{row_h - 18}" '
                f'rx="6" fill="{fill}"/>'
            )
            formatted = f"{value:+.1f}{unit}" if unit else f"{value:+.0f}"
            p.append(
                text(
                    x_cell + col_w / 2,
                    yy + 42,
                    formatted,
                    size=13,
                    fill=value_color,
                    weight=750,
                    anchor="middle",
                )
            )
            x_cell += col_w
        p.append(multiline(x_cell + 12, yy + 27, readouts[scenario], size=11, fill=TEXT, weight=650, leading=1.25))

    s5, s6 = r2["E5"]["summary"], r2["E6"]["summary"]
    max_debt_e5 = max(float(row["government_debt_ratio"]) for row in r2["E5"]["monthly"])
    max_debt_e6 = max(float(row["government_debt_ratio"]) for row in r2["E6"]["monthly"])
    e6_vs_e5_consumption = pct(
        s6["cumulative_real_consumption"], s5["cumulative_real_consumption"]
    )
    note_y = grid_y + header_h + row_h * len(policies) + 35
    p += [
        text(
            64,
            note_y,
            f"E6 vs E5: peak debt {100 * max_debt_e5:.2f}% to {100 * max_debt_e6:.2f}%; "
            f"consumption {e6_vs_e5_consumption:+.2f}%; ending firms {s5['ending_firm_count']:.0f} to {s6['ending_firm_count']:.0f}.",
            size=12,
            fill=WARNING,
            weight=700,
        ),
        text(
            64,
            note_y + 27,
            "Cells report E2-E6 relative to E1 for a common visual scale; green is improvement and red is deterioration. "
            "E6's fiscal effect is identified against E5.",
            size=10,
            fill=SECONDARY,
        ),
        text(
            64,
            note_y + 50,
            "R2 main experiment; N=500; months 25-120; seed=1. Descriptive mechanism evidence, not a robust policy ranking.",
            size=10,
            fill=SECONDARY,
        ),
    ]
    save_svg("figure_2_r2_policy_outcomes.svg", p)


def figure_mechanisms(runs: dict[str, dict[str, dict]]) -> None:
    r2 = runs["R2"]
    s2, s3, s4, s5, s6 = [r2[e]["summary"] for e in ("E2", "E3", "E4", "E5", "E6")]
    width, height = 1500, 950
    p = svg_start(
        width,
        height,
        "Supplementary Figure S2. Mechanism activation and leakage",
        "Four panels show the employment, levy, solo-enterprise, and integrated policy channels in R2.",
    )
    p += [
        text(64, 64, "SUPPLEMENTARY FIGURE S2  |  MECHANISM AUDIT", size=14, fill=FOCUS, weight=700),
        text(64, 108, "The mechanisms activate—but benefits leak through other margins", size=32, weight=700),
        text(
            64,
            139,
            "Direct policy outputs are paired with the displacement, exit, or fiscal channel that limits their welfare effect.",
            size=16,
            fill=SECONDARY,
        ),
    ]
    cards = [(64, 190), (770, 190), (64, 555), (770, 555)]
    card_w, card_h = 666, 315
    for x, y in cards:
        p.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="14" fill="{SURFACE}"/>')

    x, y = cards[0]
    p += [
        text(x + 24, y + 34, "E2  ·  Employment responsibility", size=18, weight=750),
        text(x + 24, y + 62, "Inside-firm retention works; firm-exit risk remains.", size=13, fill=SECONDARY),
    ]
    e2_metrics = [
        ("AI layoffs blocked", s2["cumulative_ai_attributable_layoffs_blocked"], POSITIVE),
        ("Exit-job losses", s2["cumulative_firm_exit_layoffs"], NEGATIVE),
        ("Entry jobs", s2["cumulative_entry_jobs"], WARNING),
    ]
    max_e2 = max(v for _, v, _ in e2_metrics)
    for i, (label, value, color) in enumerate(e2_metrics):
        yy = y + 112 + i * 52
        p.append(text(x + 24, yy, label, size=12, fill=SECONDARY))
        p.append(f'<rect x="{x + 170}" y="{yy - 14}" width="390" height="18" rx="4" fill="{GRID}"/>')
        p.append(f'<rect x="{x + 170}" y="{yy - 14}" width="{390 * value / max_e2:.1f}" height="18" rx="4" fill="{color}"/>')
        p.append(text(x + 580, yy, f"{value:,.0f}", size=13, fill=color, weight=750))
    p.append(text(x + 24, y + 285, f"Required hours fall to {s2['tail_average_required_work_hours']:.1f}; ending firms fall to {s2['ending_firm_count']}.", size=12))

    x, y = cards[1]
    p += [
        text(x + 24, y + 34, "E3  ·  AI levy and social return", size=18, weight=750),
        text(x + 24, y + 62, "The earmarked fund closes its accounting identity.", size=13, fill=SECONDARY),
    ]
    levy = s3["cumulative_ai_levy_revenue"]
    service = s3["cumulative_ai_levy_public_service_spending"]
    investment = s3["cumulative_ai_levy_public_investment"]
    balance = s3["ending_ai_levy_fund_balance"]
    stack_x, stack_y, stack_w = x + 24, y + 105, 610
    p.append(f'<rect x="{stack_x}" y="{stack_y}" width="{stack_w}" height="46" rx="8" fill="{GRID}"/>')
    p.append(f'<rect x="{stack_x}" y="{stack_y}" width="{stack_w * service / levy:.1f}" height="46" rx="8" fill="{FOCUS}"/>')
    p.append(f'<rect x="{stack_x + stack_w * service / levy:.1f}" y="{stack_y}" width="{stack_w * investment / levy:.1f}" height="46" fill="{POSITIVE}"/>')
    p += [
        text(stack_x + 12, stack_y + 29, f"Public service  {service / levy:.1%}", size=12, weight=750),
        text(stack_x + stack_w - 12, stack_y + 29, f"Investment  {investment / levy:.1%}", size=12, fill=BG, weight=750, anchor="end"),
        text(x + 24, y + 190, f"Levy revenue", size=12, fill=SECONDARY),
        text(x + 24, y + 222, f"{levy / 1e6:.2f}m", size=30, fill=FOCUS, weight=800),
        text(x + 250, y + 190, "Ending fund", size=12, fill=SECONDARY),
        text(x + 250, y + 222, f"{balance / 1000:.1f}k", size=30, fill=POSITIVE, weight=800),
        text(x + 24, y + 285, f"Tail public-service index: {s3['tail_public_service_index']:.2f} (+60.4% vs E1).", size=12),
    ]

    x, y = cards[2]
    p += [
        text(x + 24, y + 34, "E4  ·  Time dividend and solo enterprise", size=18, weight=750),
        text(x + 24, y + 62, "New activity is real, but most sales displace incumbents.", size=13, fill=SECONDARY),
    ]
    new_share = s4["cumulative_solo_net_additional_demand"] / s4["cumulative_solo_enterprise_sales"]
    disp_share = s4["cumulative_solo_incumbent_displacement"] / s4["cumulative_solo_enterprise_sales"]
    p.append(f'<rect x="{x + 24}" y="{y + 104}" width="610" height="54" rx="8" fill="{GRID}"/>')
    p.append(f'<rect x="{x + 24}" y="{y + 104}" width="{610 * new_share:.1f}" height="54" rx="8" fill="{POSITIVE}"/>')
    p += [
        text(x + 36, y + 137, f"New demand  {new_share:.1%}", size=13, fill=BG, weight=800),
        text(x + 610, y + 137, f"Incumbent displacement  {disp_share:.1%}", size=13, weight=750, anchor="end"),
        text(x + 24, y + 207, "Entries / exits", size=12, fill=SECONDARY),
        text(x + 24, y + 244, f"{s4['cumulative_solo_entries']:.0f} / {s4['cumulative_solo_exits']:.0f}", size=27, fill=FOCUS, weight=800),
        text(x + 258, y + 207, "Solo sales", size=12, fill=SECONDARY),
        text(x + 258, y + 244, f"{s4['cumulative_solo_enterprise_sales'] / 1e6:.2f}m", size=27, fill=FOCUS, weight=800),
        text(x + 24, y + 285, "Consumption falls 3.0% vs E1 despite lower liquidity vulnerability.", size=12),
    ]

    x, y = cards[3]
    p += [
        text(x + 24, y + 34, "E5–E6  ·  Integrated compact", size=18, weight=750),
        text(x + 24, y + 62, "Fiscal discipline changes debt exposure, not the core firm-exit trade-off.", size=13, fill=SECONDARY),
    ]
    labels = ["Blocked layoffs", "Levy (m)", "Solo entries", "Ending firms"]
    vals5 = [s5["cumulative_ai_attributable_layoffs_blocked"], s5["cumulative_ai_levy_revenue"] / 1e6, s5["cumulative_solo_entries"], s5["ending_firm_count"]]
    vals6 = [s6["cumulative_ai_attributable_layoffs_blocked"], s6["cumulative_ai_levy_revenue"] / 1e6, s6["cumulative_solo_entries"], s6["ending_firm_count"]]
    for i, (label, v5, v6) in enumerate(zip(labels, vals5, vals6)):
        xx = x + 24 + i * 150
        p.append(text(xx, y + 112, label, size=11, fill=SECONDARY))
        p.append(text(xx, y + 151, f"E5  {v5:.2f}" if i == 1 else f"E5  {v5:.0f}", size=22, fill=FOCUS, weight=800))
        p.append(text(xx, y + 178, f"E6  {v6:.2f}" if i == 1 else f"E6  {v6:.0f}", size=12, fill=SECONDARY, weight=700))
    max_debt_e5 = max(float(row["government_debt_ratio"]) for row in runs["R2"]["E5"]["monthly"])
    max_debt_e6 = max(float(row["government_debt_ratio"]) for row in runs["R2"]["E6"]["monthly"])
    p += [
        text(x + 24, y + 228, "Peak debt ratio", size=12, fill=SECONDARY),
        text(x + 24, y + 264, f"E5  {max_debt_e5 * 100:.2f}%   →   E6  {max_debt_e6 * 100:.2f}%", size=25, fill=WARNING, weight=800),
        text(x + 24, y + 293, "Both end with only 16–17 firms; tighter rules do not restore market structure.", size=12),
    ]
    save_svg("supplementary_figure_s2_mechanism_audit.svg", p)


def _trajectory(run: dict) -> list[tuple[int, float]]:
    points = []
    for row in run["monthly"]:
        month = int(float(row["month"]))
        if month >= 25:
            points.append((month, 100.0 * float(row["unemployment_rate"])))
    return points


def line_panel(
    p: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    scenario: str,
    r1: list[tuple[int, float]],
    r2: list[tuple[int, float]],
) -> None:
    p.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{SURFACE}"/>')
    p.append(text(x + 20, y + 32, f"{scenario}  ·  {SCENARIO_NAMES[scenario]}", size=16, weight=750))
    chart_x, chart_y = x + 58, y + 70
    chart_w, chart_h = width - 88, height - 118
    ymax = max(v for _, v in r1 + r2) * 1.12
    ymax = max(ymax, 8.0)
    for tick in range(0, int(ymax) + 1, 4):
        yy = chart_y + chart_h * (1 - tick / ymax)
        p.append(f'<line x1="{chart_x}" y1="{yy:.1f}" x2="{chart_x + chart_w}" y2="{yy:.1f}" stroke="{GRID}"/>')
        p.append(text(chart_x - 10, yy + 4, f"{tick}%", size=10, fill=SECONDARY, anchor="end"))
    for month in (25, 48, 72, 96, 120):
        xx = chart_x + chart_w * (month - 25) / 95
        p.append(text(xx, chart_y + chart_h + 23, month, size=10, fill=SECONDARY, anchor="middle"))
    for series, color, label in ((r1, NEUTRAL, "R1"), (r2, FOCUS, "R2")):
        pts = " ".join(
            f"{chart_x + chart_w * (month - 25) / 95:.1f},{chart_y + chart_h * (1 - value / ymax):.1f}"
            for month, value in series
        )
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        peak_month, peak_value = max(series, key=lambda item: item[1])
        px = chart_x + chart_w * (peak_month - 25) / 95
        py = chart_y + chart_h * (1 - peak_value / ymax)
        p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
        p.append(text(px + 8, py - 8, f"{label} {peak_value:.1f}% · m{peak_month}", size=11, fill=color, weight=750))


def figure_architecture(runs: dict[str, dict[str, dict]]) -> None:
    width, height = 1500, 1000
    p = svg_start(
        width,
        height,
        "Supplementary Figure S3. R1 versus R2 architecture sensitivity",
        "Unemployment trajectories and firm action distributions compare government-only LLM R1 with firm-and-government LLM R2 under E5 and E6.",
    )
    p += [
        text(64, 64, "SUPPLEMENTARY FIGURE S3  |  COGNITIVE ARCHITECTURE", size=14, fill=FOCUS, weight=700),
        text(64, 108, "Similar totals can hide architecture-dependent transition paths", size=32, weight=700),
        text(
            64,
            139,
            "R1 uses rule-based firms; R2 activates firm LLMs. E5/E6 reveal the largest timing and exit divergence.",
            size=16,
            fill=SECONDARY,
        ),
    ]
    line_panel(p, 64, 190, 680, 390, "E5", _trajectory(runs["R1"]["E5"]), _trajectory(runs["R2"]["E5"]))
    line_panel(p, 776, 190, 660, 390, "E6", _trajectory(runs["R1"]["E6"]), _trajectory(runs["R2"]["E6"]))

    p.append(f'<rect x="64" y="620" width="1372" height="290" rx="12" fill="{SURFACE}"/>')
    p.append(text(86, 655, "Firm action mix across the full post-shock period", size=17, weight=750))
    p.append(text(86, 680, "Share among classified firm actions; R2 shifts sharply from aggressive to baseline/patient.", size=12, fill=SECONDARY))
    rows = [("E5 · R1", "R1", "E5"), ("E5 · R2", "R2", "E5"), ("E6 · R1", "R1", "E6"), ("E6 · R2", "R2", "E6")]
    colors = {"aggressive": NEGATIVE, "baseline": FOCUS, "patient": POSITIVE}
    for i, (label, regime, scenario) in enumerate(rows):
        actions = runs[regime][scenario]["behavior"]["action_distributions"]["firm"]
        total = sum(actions.values())
        yy = 725 + i * 44
        p.append(text(86, yy + 14, label, size=12, fill=TEXT, weight=700))
        sx, sw = 180, 760
        cursor = sx
        for action in ("aggressive", "baseline", "patient"):
            share = actions.get(action, 0) / total
            segment = sw * share
            p.append(f'<rect x="{cursor:.1f}" y="{yy}" width="{segment:.1f}" height="24" fill="{colors[action]}"/>')
            if share > 0.08:
                p.append(text(cursor + segment / 2, yy + 17, f"{share:.0%}", size=10, fill=BG if action != "aggressive" else TEXT, weight=800, anchor="middle"))
            cursor += segment
        ending = runs[regime][scenario]["summary"]["ending_firm_count"]
        peak = 100 * runs[regime][scenario]["summary"]["peak_unemployment_rate"]
        p.append(text(970, yy + 17, f"ending firms {ending}", size=12, fill=SECONDARY))
        p.append(text(1110, yy + 17, f"peak unemployment {peak:.1f}%", size=12, fill=SECONDARY))
    p += [
        f'<rect x="1010" y="638" width="12" height="12" fill="{NEGATIVE}"/>',
        text(1029, 649, "aggressive", size=11, fill=SECONDARY),
        f'<rect x="1106" y="638" width="12" height="12" fill="{FOCUS}"/>',
        text(1125, 649, "baseline", size=11, fill=SECONDARY),
        f'<rect x="1194" y="638" width="12" height="12" fill="{POSITIVE}"/>',
        text(1213, 649, "patient", size=11, fill=SECONDARY),
        text(
            64,
            958,
            "Interpretation: aggregate consumption differs by <0.5% in E5/E6, but clustered firm exits make peak unemployment 6.6 pp higher in R2.",
            size=13,
            fill=TEXT,
        ),
    ]
    save_svg("supplementary_figure_s3_architecture_sensitivity.svg", p)


def build_data_bundle(runs: dict[str, dict[str, dict]]) -> dict:
    bundle: dict = {
        "schema_version": 1,
        "generated_from": "formal/institutional_v2",
        "design": {
            "population": 500,
            "months": 120,
            "activation_month": 25,
            "seeds": [1],
            "provider": "hkust",
            "requested_model": "gpt-3.5-turbo",
            "regimes": {
                "R1": "government LLM; firms and residents rule-driven",
                "R2": "firm and government LLM; residents rule-driven",
            },
            "scenario_names": SCENARIO_NAMES,
            "mechanism_matrix": DESIGN,
            "identification_questions": QUESTIONS,
        },
        "runs": {},
    }
    for regime in REGIMES:
        bundle["runs"][regime] = {}
        for scenario in SCENARIOS:
            run = runs[regime][scenario]
            bundle["runs"][regime][scenario] = {
                "source_folder": run["folder"],
                "source_fingerprint": run["source_fingerprint"],
                "provider": run["provider"],
                "summary": run["summary"],
                "decision_audit": {
                    "records": run["decision_audit"].get("records"),
                    "fallbacks": run["decision_audit"].get("fallbacks"),
                    "fallback_rate": run["decision_audit"].get("fallback_rate"),
                    "behavior_qualification": run["decision_audit"].get("behavior_qualification"),
                },
                "firm_action_distribution": run["behavior"]["action_distributions"].get("firm", {}),
                "unemployment_trajectory": _trajectory(run),
            }
    return bundle


def build_dashboard_spec(runs: dict[str, dict[str, dict]]) -> dict:
    r2 = runs["R2"]
    e1 = r2["E1"]["summary"]
    return {
        "eyebrow": "URBAN CUP / INSTITUTIONAL_V2 / SINGLE-SEED FORMAL RUN",
        "title": "AI 制度实验：R2 主结果与 R1 架构对照",
        "subtitle": "E0–E6 是可识别的制度机制包；R2 为企业与政府 LLM 主实验，R1 用于检验认知架构敏感性。",
        "focus": "R2",
        "source": "Formal institutional_v2; N=500; 120 months; seed=1; HKUST gpt-3.5-turbo; common activation at month 25. Descriptive, not confirmatory.",
        "kpis": [
            {"label": "正式单元", "value": "14/14", "delta": "R1 + R2; E0–E6", "tone": "positive"},
            {"label": "LLM fallback", "value": "0", "delta": "all retained cells", "tone": "positive"},
            {"label": "E3 公共服务", "value": "+60.4%", "delta": "R2 vs E1", "tone": "positive"},
            {"label": "E5 R2 失业峰值", "value": "13.8%", "delta": "+6.6 pp vs R1", "tone": "negative"},
        ],
        "charts": [
            {
                "kind": "heatmap",
                "title": "E0–E6 制度机制矩阵",
                "subtitle": "1 表示机制开启；E2/E3/E4 均以 E1 为对照",
                "span": "wide",
                "format": ".0f",
                "xLabels": ["私人AI", "保就业责任", "AI征费", "个体创业", "主动需求", "财政收紧"],
                "yLabels": SCENARIOS,
                "values": [DESIGN[e] for e in SCENARIOS],
            },
            {
                "kind": "bar",
                "title": "R2 累计实际消费",
                "subtitle": "相对 E1；单种子描述性差异",
                "unit": "%",
                "data": [
                    {
                        "label": e,
                        "value": pct(r2[e]["summary"]["cumulative_real_consumption"], e1["cumulative_real_consumption"]),
                        "highlight": e == "E3",
                    }
                    for e in SCENARIOS
                ],
            },
            {
                "kind": "bar",
                "title": "R2 公共服务指数",
                "subtitle": "尾期相对 E1",
                "unit": "%",
                "data": [
                    {
                        "label": e,
                        "value": pct(r2[e]["summary"]["tail_public_service_index"], e1["tail_public_service_index"]),
                        "highlight": e == "E3",
                    }
                    for e in SCENARIOS
                ],
            },
            {
                "kind": "dumbbell",
                "title": "R1–R2 失业峰值",
                "subtitle": "认知架构差异；E5/E6 最明显",
                "startLabel": "R1",
                "endLabel": "R2",
                "unit": "%",
                "data": [
                    {
                        "label": e,
                        "start": 100 * runs["R1"][e]["summary"]["peak_unemployment_rate"],
                        "end": 100 * runs["R2"][e]["summary"]["peak_unemployment_rate"],
                        "highlight": e in {"E5", "E6"},
                    }
                    for e in SCENARIOS
                ],
            },
            {
                "kind": "dumbbell",
                "title": "R1–R2 期末企业数",
                "subtitle": "R2 在综合政策下退出更多",
                "startLabel": "R1",
                "endLabel": "R2",
                "unit": "",
                "data": [
                    {
                        "label": e,
                        "start": runs["R1"][e]["summary"]["ending_firm_count"],
                        "end": runs["R2"][e]["summary"]["ending_firm_count"],
                        "highlight": e in {"E5", "E6"},
                    }
                    for e in SCENARIOS
                ],
            },
        ],
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    figure_framework()
    figure_design_matrix()
    figure_r2_outcomes(runs)
    figure_mechanisms(runs)
    figure_architecture(runs)
    bundle = build_data_bundle(runs)
    (FIGURES / "paper_results_data.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (FIGURES / "dashboard_spec.json").write_text(
        json.dumps(build_dashboard_spec(runs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated figures and data in {FIGURES}")


if __name__ == "__main__":
    main()
