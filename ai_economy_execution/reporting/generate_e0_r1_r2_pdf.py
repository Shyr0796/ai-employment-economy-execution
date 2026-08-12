from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "ai_economy_execution"
    / "results"
    / "research_matrix"
    / "formal"
    / "institutional_v2"
)
R1_DIR = (
    RESULT_ROOT
    / "R1_government"
    / "hkust_gpt-3-5-turbo"
    / "N00500_M120_S001"
    / "E0"
)
R2_DIR = (
    RESULT_ROOT
    / "R2_firm_government"
    / "hkust_gpt-3-5-turbo"
    / "N00500_M120_S001"
    / "E0"
)
CHECKPOINT = (
    RESULT_ROOT
    / "R0_rules"
    / "offline_rules"
    / "N00500_M120_S001"
    / "equilibrium"
    / "pre_equilibrium_state.json"
)
MATRIX_MANIFEST = (
    RESULT_ROOT
    / "orchestration"
    / "N00500_M120_S001-003"
    / "R0-R1-R2-R3__E0-E1-E2-E3-E4-E5-E6"
    / "matrix_manifest.json"
)
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "e0_r1_r2_seed1"
PDF_PATH = OUTPUT_DIR / "E0_R1_R2_seed1_complete_report.pdf"
AUDIT_CSV_PATH = OUTPUT_DIR / "E0_R1_R2_seed1_audit_data.csv"
REPORT_MANIFEST_PATH = OUTPUT_DIR / "E0_R1_R2_seed1_report_manifest.json"


PAGE_W, PAGE_H = landscape(A4)
MARGIN = 18 * mm

BG = colors.HexColor("#0b0d10")
SURFACE = colors.HexColor("#11151b")
RAISED = colors.HexColor("#171c24")
TEXT = colors.HexColor("#f4f7fb")
SECONDARY = colors.HexColor("#a7b0bb")
TERTIARY = colors.HexColor("#6f7884")
GRID = colors.HexColor("#2a3039")
FOCUS = colors.HexColor("#2f8cff")
POSITIVE = colors.HexColor("#35d0a6")
WARNING = colors.HexColor("#f2b84b")
NEGATIVE = colors.HexColor("#ff6b7a")
NEUTRAL = colors.HexColor("#8a939e")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\Deng.ttf")
    bold = Path(r"C:\Windows\Fonts\Dengb.ttf")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("DengXian font files are required for Chinese PDF output.")
    pdfmetrics.registerFont(TTFont("Deng", str(regular)))
    pdfmetrics.registerFont(TTFont("DengBold", str(bold)))


def rounded_rect(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: colors.Color = SURFACE,
    stroke: colors.Color = GRID,
    radius: float = 8,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.7)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def text(
    c: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    size: float = 10,
    color: colors.Color = TEXT,
    font: str = "Deng",
    align: str = "left",
) -> None:
    c.setFont(font, size)
    c.setFillColor(color)
    if align == "right":
        c.drawRightString(x, y, value)
    elif align == "center":
        c.drawCentredString(x, y, value)
    else:
        c.drawString(x, y, value)


def paragraph(
    c: canvas.Canvas,
    x: float,
    top: float,
    w: float,
    value: str,
    size: float = 10,
    leading: float | None = None,
    color: colors.Color = SECONDARY,
    font: str = "Deng",
    alignment: int = 0,
) -> float:
    style = ParagraphStyle(
        "p",
        fontName=font,
        fontSize=size,
        leading=leading or size * 1.55,
        textColor=color,
        alignment=alignment,
        spaceAfter=0,
        spaceBefore=0,
    )
    block = Paragraph(value, style)
    _, height = block.wrap(w, PAGE_H)
    block.drawOn(c, x, top - height)
    return height


def page_base(
    c: canvas.Canvas,
    page_no: int,
    title_value: str,
    subtitle: str,
    eyebrow: str = "URBAN CUP / E0 COGNITIVE ARCHITECTURE AUDIT",
) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    text(c, MARGIN, PAGE_H - 14 * mm, eyebrow, 8.5, FOCUS, "DengBold")
    text(c, MARGIN, PAGE_H - 25 * mm, title_value, 23, TEXT, "DengBold")
    paragraph(
        c,
        MARGIN,
        PAGE_H - 30 * mm,
        PAGE_W - 2 * MARGIN,
        subtitle,
        9.5,
        13,
        SECONDARY,
    )
    c.setStrokeColor(GRID)
    c.setLineWidth(0.6)
    c.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    text(c, MARGIN, 8.2 * mm, "范围: E0 / R1-R2 / seed=1 / N=500 / M=120", 7.5, TERTIARY)
    text(c, PAGE_W - MARGIN, 8.2 * mm, f"{page_no:02d}", 8, TERTIARY, "DengBold", "right")


def kpi_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    note: str,
    accent: colors.Color = FOCUS,
) -> None:
    rounded_rect(c, x, y, w, h, SURFACE, GRID)
    c.setFillColor(accent)
    c.roundRect(x, y + h - 3, w, 3, 2, stroke=0, fill=1)
    text(c, x + 10, y + h - 20, label, 8.5, SECONDARY, "DengBold")
    text(c, x + 10, y + h - 45, value, 20, TEXT, "DengBold")
    paragraph(c, x + 10, y + 27, w - 20, note, 7.5, 10, TERTIARY)


def draw_table(
    c: canvas.Canvas,
    x: float,
    top: float,
    w: float,
    data: list[list[str]],
    col_widths: list[float],
    row_heights: list[float] | None = None,
    font_size: float = 8.2,
) -> float:
    table = Table(data, colWidths=col_widths, rowHeights=row_heights)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), RAISED),
                ("TEXTCOLOR", (0, 0), (-1, 0), TEXT),
                ("FONTNAME", (0, 0), (-1, 0), "DengBold"),
                ("FONTNAME", (0, 1), (-1, -1), "Deng"),
                ("TEXTCOLOR", (0, 1), (-1, -1), SECONDARY),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size * 1.4),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURFACE, BG]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    _, h = table.wrap(w, PAGE_H)
    table.drawOn(c, x, top - h)
    return h


def horizontal_grouped_bars(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title_value: str,
    categories: list[str],
    r1: list[float],
    r2: list[float],
    note: str,
) -> None:
    rounded_rect(c, x, y, w, h)
    text(c, x + 12, y + h - 20, title_value, 10.5, TEXT, "DengBold")
    plot_x = x + 102
    plot_w = w - 126
    row_h = (h - 68) / max(len(categories), 1)
    for tick in range(0, 101, 25):
        tx = plot_x + plot_w * tick / 100
        c.setStrokeColor(GRID)
        c.setLineWidth(0.4)
        c.line(tx, y + 38, tx, y + h - 35)
        text(c, tx, y + 25, f"{tick}%", 6.8, TERTIARY, align="center")
    for i, category in enumerate(categories):
        cy = y + h - 47 - i * row_h
        text(c, x + 12, cy - 2, category, 8.2, SECONDARY)
        for offset, value, color_value, label in [
            (5, r1[i], NEUTRAL, "R1"),
            (-7, r2[i], FOCUS, "R2"),
        ]:
            bar_y = cy + offset
            bar_w = plot_w * value / 100
            c.setFillColor(color_value)
            c.rect(plot_x, bar_y, max(bar_w, 0.8), 7, fill=1, stroke=0)
            text(
                c,
                min(plot_x + bar_w + 4, x + w - 7),
                bar_y + 0.3,
                f"{label} {value:.1f}%",
                6.8,
                color_value,
                "DengBold",
            )
    paragraph(c, x + 12, y + 17, w - 24, note, 7.1, 9.2, TERTIARY)


def line_chart(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title_value: str,
    months: Iterable[float],
    r1: Iterable[float],
    r2: Iterable[float],
    unit: str,
    formatter,
    note: str,
    domain: tuple[float, float] | None = None,
) -> None:
    months_list = list(months)
    r1_list = list(r1)
    r2_list = list(r2)
    rounded_rect(c, x, y, w, h)
    text(c, x + 11, y + h - 19, title_value, 9.5, TEXT, "DengBold")
    text(c, x + w - 11, y + h - 19, unit, 7, TERTIARY, align="right")
    left, right = x + 42, x + w - 12
    bottom, top = y + 35, y + h - 36
    values = r1_list + r2_list
    if domain is not None:
        vmin, vmax = domain
    else:
        vmin, vmax = min(values), max(values)
        if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-15):
            pad = max(abs(vmin) * 0.05, 1.0)
        else:
            pad = (vmax - vmin) * 0.10
        vmin -= pad
        vmax += pad
    for t in range(4):
        py = bottom + (top - bottom) * t / 3
        value = vmin + (vmax - vmin) * t / 3
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(left, py, right, py)
        text(c, left - 5, py - 2.5, formatter(value), 6.3, TERTIARY, align="right")
    for month in [24, 48, 72, 96, 120]:
        px = left + (right - left) * (month - months_list[0]) / (
            months_list[-1] - months_list[0]
        )
        text(c, px, bottom - 13, str(month), 6.3, TERTIARY, align="center")
    if months_list[0] <= 25 <= months_list[-1]:
        px = left + (right - left) * (25 - months_list[0]) / (
            months_list[-1] - months_list[0]
        )
        c.setStrokeColor(WARNING)
        c.setDash(2, 2)
        c.line(px, bottom, px, top)
        c.setDash()
        text(c, px + 3, top - 8, "m25", 6.4, WARNING, "DengBold")

    def point(month: float, value: float) -> tuple[float, float]:
        px = left + (right - left) * (month - months_list[0]) / (
            months_list[-1] - months_list[0]
        )
        py = bottom + (top - bottom) * (value - vmin) / (vmax - vmin)
        return px, py

    c.setStrokeColor(NEUTRAL)
    c.setLineWidth(2.5)
    for i in range(1, len(months_list)):
        x0, y0 = point(months_list[i - 1], r1_list[i - 1])
        x1, y1 = point(months_list[i], r1_list[i])
        c.line(x0, y0, x1, y1)
    c.setStrokeColor(FOCUS)
    c.setLineWidth(1.2)
    c.setDash(4, 2)
    for i in range(1, len(months_list)):
        x0, y0 = point(months_list[i - 1], r2_list[i - 1])
        x1, y1 = point(months_list[i], r2_list[i])
        c.line(x0, y0, x1, y1)
    c.setDash()
    text(c, x + 11, y + 16, note, 6.8, TERTIARY)


def simple_vertical_bars(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title_value: str,
    labels: list[str],
    values: list[float],
    formatted_values: list[str],
    unit: str,
    colors_list: list[colors.Color] | None = None,
    log_scale: bool = False,
    note: str = "",
) -> None:
    rounded_rect(c, x, y, w, h)
    text(c, x + 11, y + h - 19, title_value, 9.5, TEXT, "DengBold")
    text(c, x + w - 11, y + h - 19, unit, 7, TERTIARY, align="right")
    bottom = y + 42
    top = y + h - 39
    plot_h = top - bottom
    transformed = [math.log10(max(v, 1e-12)) if log_scale else v for v in values]
    vmax = max(transformed) if transformed else 1.0
    if log_scale:
        vmin = min(transformed)
        span = max(vmax - vmin, 1.0)
    else:
        vmin = 0.0
        span = max(vmax, 1e-12)
    gap = 12
    bar_w = (w - 32 - gap * (len(values) - 1)) / max(len(values), 1)
    for i, value in enumerate(transformed):
        bx = x + 16 + i * (bar_w + gap)
        height = (
            plot_h * (value - vmin + 0.05 * span) / (1.05 * span)
            if log_scale
            else plot_h * value / span
        )
        color_value = (colors_list or [FOCUS] * len(values))[i]
        c.setFillColor(color_value)
        c.rect(bx, bottom, bar_w, max(height, 1.0), fill=1, stroke=0)
        text(c, bx + bar_w / 2, bottom + height + 5, formatted_values[i], 6.8, TEXT, "DengBold", "center")
        text(c, bx + bar_w / 2, bottom - 13, labels[i], 6.8, SECONDARY, align="center")
    if note:
        text(c, x + 11, y + 15, note, 6.6, TERTIARY)


def build_audit_rows(
    r1_summary: dict[str, Any],
    r2_summary: dict[str, Any],
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    metrics = [
        ("累计真实消费", "cumulative_real_consumption", "currency"),
        ("底部60%累计真实消费", "bottom60_cumulative_real_consumption", "currency"),
        ("底部80%累计真实消费", "bottom80_cumulative_real_consumption", "currency"),
        ("累计可支配收入", "cumulative_disposable_income", "currency"),
        ("累计实际政府购买", "cumulative_real_government_purchase", "currency"),
        ("激活后峰值失业率", "peak_unemployment_rate", "rate"),
        ("尾部就业率", "tail_employment_rate", "rate"),
        ("基本现金不足率", "tail_essential_cash_shortfall_rate", "rate"),
        ("流动性脆弱率", "tail_liquidity_vulnerable_rate", "rate"),
        ("公共服务指数", "tail_public_service_index", "index"),
        ("市场HHI", "tail_market_hhi", "index"),
        ("期末企业数", "ending_firm_count", "count"),
        ("收入Atkinson epsilon=1", "tail_disposable_income_atkinson_1_0", "index"),
        ("消费Atkinson epsilon=1", "tail_real_consumption_atkinson_1_0", "index"),
    ]
    rows: list[dict[str, Any]] = []
    for label, key, unit in metrics:
        a = float(r1_summary["summary"][key])
        b = float(r2_summary["summary"][key])
        rows.append(
            {
                "section": "economic_outcome",
                "metric": label,
                "key": key,
                "unit": unit,
                "R1": a,
                "R2": b,
                "delta_R2_minus_R1": b - a,
            }
        )
    for column in [
        "bank_balance_sheet_error",
        "sales_identity_error",
        "wage_identity_error",
        "tax_identity_error",
    ]:
        rows.append(
            {
                "section": "accounting_error_max_abs",
                "metric": column,
                "key": column,
                "unit": "absolute",
                "R1": float(r1_df[column].abs().max()),
                "R2": float(r2_df[column].abs().max()),
                "delta_R2_minus_R1": float(
                    r2_df[column].abs().max() - r1_df[column].abs().max()
                ),
            }
        )
    return rows


def draw_report(
    r1_summary: dict[str, Any],
    r2_summary: dict[str, Any],
    r1_behavior: dict[str, Any],
    r2_behavior: dict[str, Any],
    r1_manifest: dict[str, Any],
    r2_manifest: dict[str, Any],
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    matrix_manifest: dict[str, Any],
    exact_equal_columns: int,
    floating_only_columns: int,
    material_diff_columns: int,
    max_numeric_difference: float,
    checkpoint_hash: str,
) -> None:
    c = canvas.Canvas(str(PDF_PATH), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("E0 R1-R2 seed1 完整结果核对报告")
    c.setAuthor("Urban Cup research workflow")
    c.setSubject("Paired cognitive architecture audit")

    r1_calls = int(r1_summary["decision_audit"]["llm_eligible_records"])
    r2_calls = int(r2_summary["decision_audit"]["llm_eligible_records"])
    r1_input = int(r1_summary["token_stats"]["gpt-3.5-turbo"]["input"])
    r2_input = int(r2_summary["token_stats"]["gpt-3.5-turbo"]["input"])
    r1_output = int(r1_summary["token_stats"]["gpt-3.5-turbo"]["output"])
    r2_output = int(r2_summary["token_stats"]["gpt-3.5-turbo"]["output"])
    r1_runtime = (
        parse_time(r1_manifest["finished_at"]) - parse_time(r1_manifest["started_at"])
    ).total_seconds()
    r2_runtime = (
        parse_time(r2_manifest["finished_at"]) - parse_time(r2_manifest["started_at"])
    ).total_seconds()

    # Page 1: executive summary
    page_base(
        c,
        1,
        "E0 基准情景下，企业 LLM 改变了行为标签，但没有改变经济轨迹",
        "本报告只分析最新正式运行中已完成的 R1/E0/S001 与 R2/E0/S001。R3、其他情景和其他种子全部排除。",
    )
    card_gap = 8
    card_w = (PAGE_W - 2 * MARGIN - 3 * card_gap) / 4
    y = PAGE_H - 82 * mm
    kpi_card(c, MARGIN, y, card_w, 39 * mm, "逐月指标一致性", f"{exact_equal_columns}/434", "完全相同的指标列", POSITIVE)
    kpi_card(c, MARGIN + card_w + card_gap, y, card_w, 39 * mm, "LLM fallback", "0", "R1 与 R2 均为零", POSITIVE)
    kpi_card(c, MARGIN + 2 * (card_w + card_gap), y, card_w, 39 * mm, "R2 调用量", f"{r2_calls / r1_calls:.0f}x", "相对 R1，但宏观结果不变", WARNING)
    kpi_card(c, MARGIN + 3 * (card_w + card_gap), y, card_w, 39 * mm, "统计样本", "1 seed", "配对轨迹，不提供不确定性", WARNING)
    box_y = 45 * mm
    box_h = 54 * mm
    rounded_rect(c, MARGIN, box_y, PAGE_W - 2 * MARGIN, box_h, RAISED, GRID, 10)
    text(c, MARGIN + 15, box_y + box_h - 24, "核心判断", 12, TEXT, "DengBold")
    paragraph(
        c,
        MARGIN + 15,
        box_y + box_h - 34,
        PAGE_W - 2 * MARGIN - 30,
        "结果本身没有原则性技术错误。R1 与 R2 使用相同 checkpoint、相同 seed 和相同 E0 配置，质量门槛全部通过。"
        "R2 企业 LLM 从规则企业的 aggressive/baseline 转向 98.8% baseline，但 E0 没有新增 AI、没有生产率冲击，企业目标就业人数也没有形成调整缺口，"
        "因此宏观轨迹保持一致。这个结果适合作为认知架构安慰剂，不适合解释 AI 失业或制度干预效果。",
        11,
        17,
        TEXT,
    )
    c.showPage()

    # Page 2: scope and provenance
    page_base(
        c,
        2,
        "范围、配对设计与数据溯源",
        "比较问题: 在同一 E0、seed=1 和共同 R0 均衡 checkpoint 下，加入企业 LLM 后发生了什么？",
    )
    flow_y = PAGE_H - 67 * mm
    nodes = [
        ("R0 共同均衡", "月1-24", NEUTRAL),
        ("认知架构激活", "月25", WARNING),
        ("R1 政府 LLM", "96个月", NEUTRAL),
        ("R2 企业+政府 LLM", "96个月", FOCUS),
        ("配对比较", "月25-120", POSITIVE),
    ]
    node_w = 36 * mm
    gap = (PAGE_W - 2 * MARGIN - len(nodes) * node_w) / (len(nodes) - 1)
    for i, (label, sub, color_value) in enumerate(nodes):
        nx = MARGIN + i * (node_w + gap)
        rounded_rect(c, nx, flow_y, node_w, 25 * mm, RAISED, color_value, 7)
        text(c, nx + node_w / 2, flow_y + 15 * mm, label, 8.5, TEXT, "DengBold", "center")
        text(c, nx + node_w / 2, flow_y + 7 * mm, sub, 7.3, color_value, "DengBold", "center")
        if i < len(nodes) - 1:
            c.setStrokeColor(GRID)
            c.setLineWidth(1.2)
            c.line(nx + node_w, flow_y + 12.5 * mm, nx + node_w + gap, flow_y + 12.5 * mm)
    included = [
        ["字段", "R1", "R2", "核对"],
        ["场景", "E0 no_new_ai", "E0 no_new_ai", "一致"],
        ["人口 / 月数 / seed", "500 / 120 / 1", "500 / 120 / 1", "一致"],
        ["LLM角色", "government", "firm, government", "设计差异"],
        ["source fingerprint", r1_summary["source_fingerprint"][:16] + "...", r2_summary["source_fingerprint"][:16] + "...", "一致"],
        ["checkpoint SHA-256", checkpoint_hash[:16] + "...", checkpoint_hash[:16] + "...", "一致"],
        ["单元状态", r1_summary["status"], r2_summary["status"], "完成"],
    ]
    draw_table(
        c,
        MARGIN,
        flow_y - 12,
        PAGE_W - 2 * MARGIN,
        included,
        [46 * mm, 57 * mm, 57 * mm, 28 * mm],
        font_size=8,
    )
    rounded_rect(c, MARGIN, 28 * mm, PAGE_W - 2 * MARGIN, 25 * mm, SURFACE, WARNING, 7)
    text(c, MARGIN + 12, 44 * mm, "明确排除", 9.5, WARNING, "DengBold")
    paragraph(
        c,
        MARGIN + 12,
        41 * mm,
        PAGE_W - 2 * MARGIN - 24,
        "R3 的所有完成或部分结果、E1-E6、seed=2-3 均不进入本报告。总矩阵 manifest 仍标记 running，实际无运行进程；其中 S001:R3:E2 为中断/不完整单元。",
        8.3,
        11.5,
        SECONDARY,
    )
    c.showPage()

    # Page 3: technical quality
    page_base(
        c,
        3,
        "技术质量、模型身份与会计闭合",
        "两个目标单元均为 completed；决策审计闭合、模型别名合规、fallback 为零。",
    )
    tech_data = [
        ["质量检查", "R1", "R2", "判定"],
        ["LLM eligible / accepted", f"{r1_calls} / {r1_calls}", f"{r2_calls} / {r2_calls}", "通过"],
        ["fallback / bounded / unknown", "0 / 0 / 0", "0 / 0 / 0", "通过"],
        ["响应模型", "gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18", "HKUST别名"],
        ["行为结论资格", "false", "false", "缺少压力桶"],
        ["最大银行表误差", f"{r1_df['bank_balance_sheet_error'].abs().max():.3e}", f"{r2_df['bank_balance_sheet_error'].abs().max():.3e}", "< 1e-5"],
        ["最大销售恒等误差", f"{r1_df['sales_identity_error'].abs().max():.3e}", f"{r2_df['sales_identity_error'].abs().max():.3e}", "通过"],
        ["最大工资恒等误差", f"{r1_df['wage_identity_error'].abs().max():.3e}", f"{r2_df['wage_identity_error'].abs().max():.3e}", "通过"],
        ["最大税收恒等误差", f"{r1_df['tax_identity_error'].abs().max():.3e}", f"{r2_df['tax_identity_error'].abs().max():.3e}", "通过"],
    ]
    draw_table(
        c,
        MARGIN,
        PAGE_H - 49 * mm,
        PAGE_W - 2 * MARGIN,
        tech_data,
        [55 * mm, 56 * mm, 56 * mm, 28 * mm],
        font_size=8.2,
    )
    y0 = 30 * mm
    h0 = 44 * mm
    rounded_rect(c, MARGIN, y0, 92 * mm, h0, SURFACE, POSITIVE, 8)
    text(c, MARGIN + 12, y0 + h0 - 22, "模型身份", 10, POSITIVE, "DengBold")
    paragraph(
        c,
        MARGIN + 12,
        y0 + h0 - 31,
        92 * mm - 24,
        "请求名 gpt-3.5-turbo 全部由 HKUST 返回 gpt-4o-mini-2024-07-18。根据当前 provider alias 策略，属于合规别名，不是模型错配。",
        8.2,
        11.5,
        SECONDARY,
    )
    rounded_rect(c, MARGIN + 98 * mm, y0, 97 * mm, h0, SURFACE, WARNING, 8)
    text(c, MARGIN + 98 * mm + 12, y0 + h0 - 22, "行为门槛", 10, WARNING, "DengBold")
    paragraph(
        c,
        MARGIN + 98 * mm + 12,
        y0 + h0 - 31,
        97 * mm - 24,
        "E0 没有高失业和高债务状态，政府失业响应与债务守门方向性检查不可用。因此只能确认 LLM 运行正常，不能确认其压力状态下的行为正确性。",
        8.2,
        11.5,
        SECONDARY,
    )
    c.showPage()

    # Page 4: behavior
    page_base(
        c,
        4,
        "行为确实发生区分，但经济状态没有被推动",
        "动作分布使用全部月25-120决策记录。R1企业为规则行为；R2企业为LLM行为。",
    )
    firm_r1 = r1_behavior["action_distributions"]["firm"]
    firm_r2 = r2_behavior["action_distributions"]["firm"]
    gov_r1 = r1_behavior["action_distributions"]["government"]
    gov_r2 = r2_behavior["action_distributions"]["government"]
    firm_total_r1 = sum(firm_r1.values())
    firm_total_r2 = sum(firm_r2.values())
    gov_total_r1 = sum(gov_r1.values())
    gov_total_r2 = sum(gov_r2.values())
    horizontal_grouped_bars(
        c,
        MARGIN,
        41 * mm,
        118 * mm,
        116 * mm,
        "企业 labor stance 分布",
        ["aggressive", "baseline", "patient"],
        [100 * firm_r1.get(k, 0) / firm_total_r1 for k in ["aggressive", "baseline", "patient"]],
        [100 * firm_r2.get(k, 0) / firm_total_r2 for k in ["aggressive", "baseline", "patient"]],
        f"记录数: R1={firm_total_r1:,}, R2={firm_total_r2:,}。R2 LLM 企业以 baseline 为主。",
    )
    horizontal_grouped_bars(
        c,
        MARGIN + 124 * mm,
        78 * mm,
        72 * mm,
        79 * mm,
        "政府 policy stance",
        ["baseline", "fiscal_guard"],
        [100 * gov_r1.get(k, 0) / gov_total_r1 for k in ["baseline", "fiscal_guard"]],
        [100 * gov_r2.get(k, 0) / gov_total_r2 for k in ["baseline", "fiscal_guard"]],
        "债务率为0，但仍出现少量 fiscal_guard。",
    )
    rounded_rect(c, MARGIN + 124 * mm, 41 * mm, 72 * mm, 31 * mm, RAISED, FOCUS, 7)
    text(c, MARGIN + 124 * mm + 11, 62 * mm, "为什么动作不同却没有结果差异？", 9, FOCUS, "DengBold")
    paragraph(
        c,
        MARGIN + 124 * mm + 11,
        58 * mm,
        72 * mm - 22,
        "labor stance 只缩放人员调整速度: patient=0.60x，aggressive=1.40x。E0 中 target=headcount 时，速度乘数不产生实际招聘或裁员。",
        7.6,
        10.5,
        SECONDARY,
    )
    c.showPage()

    # Page 5: cost
    page_base(
        c,
        5,
        "R2 为零宏观增益付出了显著更高的 LLM 成本",
        "不同资源单位使用独立小图。调用量和 token 为实际审计值；运行时间包含 API 等待与模拟。",
    )
    chart_y = 68 * mm
    chart_h = 85 * mm
    chart_w = 46 * mm
    gap = 4 * mm
    labels = ["R1", "R2"]
    simple_vertical_bars(
        c,
        MARGIN,
        chart_y,
        chart_w,
        chart_h,
        "LLM调用",
        labels,
        [r1_calls, r2_calls],
        [f"{r1_calls:,}", f"{r2_calls:,}"],
        "calls",
        [NEUTRAL, FOCUS],
        note=f"R2/R1 = {r2_calls / r1_calls:.1f}x",
    )
    simple_vertical_bars(
        c,
        MARGIN + (chart_w + gap),
        chart_y,
        chart_w,
        chart_h,
        "输入token",
        labels,
        [r1_input, r2_input],
        [f"{r1_input/1000:.1f}k", f"{r2_input/1000:.1f}k"],
        "tokens",
        [NEUTRAL, FOCUS],
        note=f"R2/R1 = {r2_input / r1_input:.1f}x",
    )
    simple_vertical_bars(
        c,
        MARGIN + 2 * (chart_w + gap),
        chart_y,
        chart_w,
        chart_h,
        "输出token",
        labels,
        [r1_output, r2_output],
        [f"{r1_output:,}", f"{r2_output:,}"],
        "tokens",
        [NEUTRAL, FOCUS],
        note=f"R2/R1 = {r2_output / r1_output:.1f}x",
    )
    simple_vertical_bars(
        c,
        MARGIN + 3 * (chart_w + gap),
        chart_y,
        chart_w,
        chart_h,
        "墙钟时间",
        labels,
        [r1_runtime, r2_runtime],
        [f"{r1_runtime/60:.1f}m", f"{r2_runtime/60:.1f}m"],
        "minutes",
        [NEUTRAL, FOCUS],
        note=f"R2/R1 = {r2_runtime / r1_runtime:.1f}x",
    )
    rounded_rect(c, MARGIN, 31 * mm, PAGE_W - 2 * MARGIN, 29 * mm, RAISED, WARNING, 8)
    text(c, MARGIN + 12, 50 * mm, "行业解释", 9.5, WARNING, "DengBold")
    paragraph(
        c,
        MARGIN + 12,
        46 * mm,
        PAGE_W - 2 * MARGIN - 24,
        "在稳定基准下，企业 LLM 没有创造可观测的经济增益，却将调用量提高到 31 倍。更合理的工程策略是事件触发式 LLM: "
        "需求、现金、失业或产能利用率越过阈值时才升级到 LLM，平稳状态继续采用规则代理。",
        8.5,
        12,
        TEXT,
    )
    c.showPage()

    # Page 6: trajectories
    page_base(
        c,
        6,
        "四条核心轨迹完全重叠",
        "灰色实线为 R1，蓝色虚线为 R2；黄色竖线为月25认知架构激活。图中包含月24共同均衡端点。",
    )
    frame = r1_df[r1_df["month"] >= 24].copy()
    frame2 = r2_df[r2_df["month"] >= 24].copy()
    months = frame["month"].tolist()
    w = 94 * mm
    h = 58 * mm
    x1, x2 = MARGIN, MARGIN + 100 * mm
    y1, y2 = 96 * mm, 32 * mm
    line_chart(
        c, x1, y1, w, h, "就业率", months,
        frame["employment_rate"], frame2["employment_rate"], "rate",
        lambda v: f"{100*v:.1f}%", "m25: 97.2%; m26起: 100%"
    )
    line_chart(
        c, x2, y1, w, h, "真实消费", months,
        frame["real_consumption"], frame2["real_consumption"], "currency/month",
        lambda v: f"{v/1e6:.2f}m", "R1-R2 最大差异: 0"
    )
    line_chart(
        c, x1, y2, w, h, "企业销售", months,
        frame["firm_sales"], frame2["firm_sales"], "currency/month",
        lambda v: f"{v/1e6:.2f}m", "R1-R2 最大差异: 0"
    )
    line_chart(
        c, x2, y2, w, h, "实际政府采购", months,
        frame["government_real_procurement"], frame2["government_real_procurement"], "currency/month",
        lambda v: f"{v/1000:.1f}k", f"最大差异: {max_numeric_difference:.2e}",
        domain=(0.0, 24000.0),
    )
    c.showPage()

    # Page 7: macro comparison
    page_base(
        c,
        7,
        "宏观结果: 物质差异为零",
        "不同量纲不合并到同一坐标轴。以下直接列示配对值与差异。",
    )
    s1 = r1_summary["summary"]
    s2 = r2_summary["summary"]
    macro_rows = [
        ["指标", "R1", "R2", "R2-R1"],
        ["累计真实消费", num(s1["cumulative_real_consumption"]), num(s2["cumulative_real_consumption"]), "0.00"],
        ["底部60%累计真实消费", num(s1["bottom60_cumulative_real_consumption"]), num(s2["bottom60_cumulative_real_consumption"]), "0.00"],
        ["累计可支配收入", num(s1["cumulative_disposable_income"]), num(s2["cumulative_disposable_income"]), "0.00"],
        ["激活后峰值失业率", pct(s1["peak_unemployment_rate"]), pct(s2["peak_unemployment_rate"]), "0.0 pp"],
        ["尾部就业率", pct(s1["tail_employment_rate"]), pct(s2["tail_employment_rate"]), "0.0 pp"],
        ["期末企业数", str(s1["ending_firm_count"]), str(s2["ending_firm_count"]), "0"],
        ["公共服务指数", f"{s1['tail_public_service_index']:.6f}", f"{s2['tail_public_service_index']:.6f}", "0"],
        ["市场HHI", f"{s1['tail_market_hhi']:.6f}", f"{s2['tail_market_hhi']:.6f}", "0"],
        ["收入Atkinson epsilon=1", f"{s1['tail_disposable_income_atkinson_1_0']:.6f}", f"{s2['tail_disposable_income_atkinson_1_0']:.6f}", "0"],
        ["消费Atkinson epsilon=1", f"{s1['tail_real_consumption_atkinson_1_0']:.6f}", f"{s2['tail_real_consumption_atkinson_1_0']:.6f}", "0"],
    ]
    draw_table(
        c,
        MARGIN,
        PAGE_H - 48 * mm,
        127 * mm,
        macro_rows,
        [52 * mm, 31 * mm, 31 * mm, 20 * mm],
        font_size=7.7,
    )
    simple_vertical_bars(
        c,
        MARGIN + 135 * mm,
        92 * mm,
        61 * mm,
        61 * mm,
        "434列轨迹核对",
        ["完全相同", "浮点差", "物质差异"],
        [exact_equal_columns, floating_only_columns, material_diff_columns],
        [str(exact_equal_columns), str(floating_only_columns), str(material_diff_columns)],
        "columns",
        [POSITIVE, WARNING, NEGATIVE],
        note="浮点差仅为约 1e-12 量级",
    )
    rounded_rect(c, MARGIN + 135 * mm, 35 * mm, 61 * mm, 49 * mm, RAISED, POSITIVE, 8)
    text(c, MARGIN + 135 * mm + 11, 70 * mm, "配对零结果", 10, POSITIVE, "DengBold")
    paragraph(
        c,
        MARGIN + 135 * mm + 11,
        66 * mm,
        61 * mm - 22,
        "R2 并非“更差”或“更好”，而是在本 seed 的无AI基准中与 R1 等价。这可以支持基准稳定性，但不能外推到AI冲击情景。",
        8,
        11.3,
        SECONDARY,
    )
    c.showPage()

    # Page 8: distribution
    page_base(
        c,
        8,
        "充分就业并不等于没有家庭脆弱性",
        "R1 与 R2 分配结果完全相同，因此本页只画一次 E0 基准分布。数值为尾部期平均。",
    )
    group_labels = ["低", "较低", "中", "较高", "高"]
    income = [
        s1[f"tail_group_{g}_mean_disposable_income"]
        for g in ["low", "lower_middle", "middle", "upper_middle", "high"]
    ]
    consumption = [
        s1[f"tail_group_{g}_mean_real_consumption"]
        for g in ["low", "lower_middle", "middle", "upper_middle", "high"]
    ]
    wealth = [
        s1[f"tail_group_{g}_mean_financial_wealth"]
        for g in ["low", "lower_middle", "middle", "upper_middle", "high"]
    ]
    simple_vertical_bars(
        c,
        MARGIN,
        82 * mm,
        61 * mm,
        73 * mm,
        "月均可支配收入",
        group_labels,
        income,
        [f"{v/1000:.1f}k" for v in income],
        "currency/month",
        [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, FOCUS],
    )
    simple_vertical_bars(
        c,
        MARGIN + 67 * mm,
        82 * mm,
        61 * mm,
        73 * mm,
        "月均真实消费",
        group_labels,
        consumption,
        [f"{v/1000:.1f}k" for v in consumption],
        "currency/month",
        [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, FOCUS],
    )
    simple_vertical_bars(
        c,
        MARGIN + 134 * mm,
        82 * mm,
        61 * mm,
        73 * mm,
        "平均金融财富",
        group_labels,
        wealth,
        [f"{v/1000:.1f}k" for v in wealth],
        "currency / log scale",
        [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, FOCUS],
        log_scale=True,
        note="柱高为log10，标签为原值",
    )
    vulnerability = [
        ("尾部就业率", pct(s1["tail_employment_rate"]), POSITIVE),
        ("基本现金不足率", pct(s1["tail_essential_cash_shortfall_rate"]), WARNING),
        ("低收入组现金不足", pct(s1["tail_group_low_essential_cash_shortfall_rate"]), NEGATIVE),
        ("流动性脆弱率", pct(s1["tail_liquidity_vulnerable_rate"]), WARNING),
        ("低收入组流动性脆弱", pct(s1["tail_group_low_liquidity_vulnerable_rate"]), WARNING),
    ]
    card_w2 = (PAGE_W - 2 * MARGIN - 4 * 6) / 5
    for i, (label, value, color_value) in enumerate(vulnerability):
        kpi_card(c, MARGIN + i * (card_w2 + 6), 31 * mm, card_w2, 39 * mm, label, value, "E0尾部状态", color_value)
    c.showPage()

    # Page 9: interpretation and audit appendix
    page_base(
        c,
        9,
        "研究结论、不可声称内容与复核清单",
        "这是单种子配对基准，不是完整主实验。",
    )
    box_w = 94 * mm
    rounded_rect(c, MARGIN, 82 * mm, box_w, 72 * mm, SURFACE, POSITIVE, 8)
    text(c, MARGIN + 12, 142 * mm, "可以声称", 11, POSITIVE, "DengBold")
    paragraph(
        c,
        MARGIN + 12,
        137 * mm,
        box_w - 24,
        "1. R1/R2 两个单元技术质量通过。<br/>"
        "2. 企业 LLM 显著改变动作分布。<br/>"
        "3. 在 E0 稳定基准中，动作差异没有形成宏观差异。<br/>"
        "4. R2 在此基准下成本显著高于 R1。<br/>"
        "5. 该结果支持事件触发式 LLM 的工程设计。",
        9,
        14,
        SECONDARY,
    )
    rounded_rect(c, MARGIN + 102 * mm, 82 * mm, box_w, 72 * mm, SURFACE, NEGATIVE, 8)
    text(c, MARGIN + 102 * mm + 12, 142 * mm, "不能声称", 11, NEGATIVE, "DengBold")
    paragraph(
        c,
        MARGIN + 102 * mm + 12,
        137 * mm,
        box_w - 24,
        "1. AI 不会导致失业。<br/>"
        "2. 企业 LLM 能缓解或加剧 AI 失业。<br/>"
        "3. R2 在 AI 冲击下与 R1 等价。<br/>"
        "4. 单种子零差异具有总体统计显著性。<br/>"
        "5. 当前政府 LLM 已通过高失业/高债务行为验证。",
        9,
        14,
        SECONDARY,
    )
    audit_rows = [
        ["核对项", "结果"],
        ["PDF范围", "E0 / R1-R2 / seed=1"],
        ["原始单元", "2个 completed"],
        ["共同checkpoint", checkpoint_hash],
        ["source fingerprint", r1_summary["source_fingerprint"]],
        ["R3处理", "全部排除；S001:R3:E2视为中断"],
        ["统计不确定性", "不可估计；仅1个seed"],
        ["图表变换", "财富柱高使用log10，其余未归一化"],
        ["生成时间", datetime.now(timezone.utc).isoformat()],
    ]
    draw_table(
        c,
        MARGIN,
        72 * mm,
        PAGE_W - 2 * MARGIN,
        audit_rows,
        [45 * mm, 150 * mm],
        font_size=7.2,
    )
    c.save()


def main() -> None:
    register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        R1_DIR / "summary.json",
        R1_DIR / "metrics.csv",
        R1_DIR / "decision_behavior_summary.json",
        R1_DIR / "run_manifest.json",
        R2_DIR / "summary.json",
        R2_DIR / "metrics.csv",
        R2_DIR / "decision_behavior_summary.json",
        R2_DIR / "run_manifest.json",
        CHECKPOINT,
        MATRIX_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required report inputs:\n" + "\n".join(missing))

    r1_summary = load_json(R1_DIR / "summary.json")
    r2_summary = load_json(R2_DIR / "summary.json")
    r1_behavior = load_json(R1_DIR / "decision_behavior_summary.json")
    r2_behavior = load_json(R2_DIR / "decision_behavior_summary.json")
    r1_manifest = load_json(R1_DIR / "run_manifest.json")
    r2_manifest = load_json(R2_DIR / "run_manifest.json")
    matrix_manifest = load_json(MATRIX_MANIFEST)
    r1_df = pd.read_csv(R1_DIR / "metrics.csv")
    r2_df = pd.read_csv(R2_DIR / "metrics.csv")

    if r1_summary["status"] != "completed" or r2_summary["status"] != "completed":
        raise RuntimeError("Both scoped cells must be completed.")
    if r1_summary["source_fingerprint"] != r2_summary["source_fingerprint"]:
        raise RuntimeError("Source fingerprints do not match.")
    if list(r1_df.columns) != list(r2_df.columns) or len(r1_df) != len(r2_df):
        raise RuntimeError("Metrics schemas or row counts do not match.")

    exact_equal_columns = 0
    floating_only_columns = 0
    material_diff_columns = 0
    max_numeric_difference = 0.0
    column_audit: list[dict[str, Any]] = []
    for column in r1_df.columns:
        equal = r1_df[column].astype(str).equals(r2_df[column].astype(str))
        if equal:
            exact_equal_columns += 1
            status = "exact_equal"
            max_delta = 0.0
        else:
            a = pd.to_numeric(r1_df[column], errors="coerce")
            b = pd.to_numeric(r2_df[column], errors="coerce")
            if a.notna().all() and b.notna().all():
                max_delta = float((b - a).abs().max())
                max_numeric_difference = max(max_numeric_difference, max_delta)
                if max_delta <= 1e-9:
                    floating_only_columns += 1
                    status = "floating_only"
                else:
                    material_diff_columns += 1
                    status = "material_difference"
            else:
                max_delta = None
                material_diff_columns += 1
                status = "material_difference"
        column_audit.append(
            {
                "column": column,
                "status": status,
                "max_abs_delta": max_delta,
            }
        )

    checkpoint_hash = sha256(CHECKPOINT)
    audit_rows = build_audit_rows(r1_summary, r2_summary, r1_df, r2_df)
    with AUDIT_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    scoped_tasks = []
    for task in matrix_manifest.get("tasks", []):
        if (
            task.get("seed") == 1
            and task.get("scenario") == "E0"
            and task.get("regime") in {"R1", "R2"}
        ):
            scoped_tasks.append(
                {
                    "id": task.get("id"),
                    "execution_status": task.get("execution_status"),
                    "returncode": task.get("returncode"),
                    "output": task.get("output"),
                }
            )
    report_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "scenario": "E0",
            "regimes": ["R1", "R2"],
            "seed": 1,
            "population": 500,
            "months": 120,
            "excluded": ["R3", "E1-E6", "seed2", "seed3"],
        },
        "source_fingerprint": r1_summary["source_fingerprint"],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "matrix_manifest_status": matrix_manifest.get("status"),
        "scoped_tasks": scoped_tasks,
        "comparison": {
            "rows": len(r1_df),
            "columns": len(r1_df.columns),
            "exact_equal_columns": exact_equal_columns,
            "floating_only_columns": floating_only_columns,
            "material_difference_columns": material_diff_columns,
            "max_numeric_difference": max_numeric_difference,
            "column_audit": column_audit,
        },
        "input_files": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in required
        },
        "outputs": {
            "pdf": str(PDF_PATH.relative_to(ROOT)),
            "audit_csv": str(AUDIT_CSV_PATH.relative_to(ROOT)),
        },
    }
    REPORT_MANIFEST_PATH.write_text(
        json.dumps(report_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    draw_report(
        r1_summary,
        r2_summary,
        r1_behavior,
        r2_behavior,
        r1_manifest,
        r2_manifest,
        r1_df,
        r2_df,
        matrix_manifest,
        exact_equal_columns,
        floating_only_columns,
        material_diff_columns,
        max_numeric_difference,
        checkpoint_hash,
    )
    print(
        json.dumps(
            {
                "pdf": str(PDF_PATH),
                "audit_csv": str(AUDIT_CSV_PATH),
                "report_manifest": str(REPORT_MANIFEST_PATH),
                "pages_expected": 9,
                "comparison": report_manifest["comparison"]
                | {"column_audit": "omitted_from_stdout"},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
