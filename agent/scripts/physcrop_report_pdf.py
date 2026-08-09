"""Professional Chinese PDF export for PhysCrop-Risk reports."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

STATUS_ZH = {
    "needs_review": "待审核",
    "approved": "已批准",
    "rejected": "已驳回",
    "needs_revision": "待修改",
}
GENERATOR_ZH = {"deterministic_fallback": "规则生成", "llm_guarded": "智能润色"}
PROVIDER_ZH = {"local": "规则引擎", "deepseek": "深度求索", "openai": "OpenAI", "anthropic": "Anthropic"}


def _style(name: str, size: float = 10.5, leading: float = 16, bold: bool = False,
           color: Any = colors.HexColor("#1c2a22")) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName="STSong-Light",
        fontSize=size,
        leading=leading,
        textColor=color,
        spaceAfter=4,
    )


TITLE = _style("title", 20, 28, color=colors.HexColor("#145c3a"))
H1 = _style("h1", 13, 18, True, colors.HexColor("#145c3a"))
BODY = _style("body")
META = _style("meta", 9, 14, color=colors.HexColor("#6a7a70"))
SMALL = _style("small", 8.5, 13, color=colors.HexColor("#8b9a90"))
CELL = _style("cell", 9, 14)
CELL_BOLD = _style("cell_bold", 9, 14, True)
NOTE = _style("note", 9.5, 15, color=colors.HexColor("#6d4c18"))


def _p(value: Any, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(str(value if value is not None else "—"), style)


def _section(story: list, title: str) -> None:
    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph(title, H1))
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#1e7a50"),
                            spaceBefore=1, spaceAfter=6))


def _bullets(story: list, items: list[Any]) -> None:
    for item in items:
        story.append(Paragraph(f"· {item}", BODY))


def build_report_pdf(report: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"低产风险预警报告 · {report.get('event_key', '')}",
        author="丰谷智警",
    )
    story: list = []
    p = report["professional_analysis"]
    provenance = report["provenance"]
    status = STATUS_ZH.get(report.get("status"), report.get("status", ""))
    status_color = colors.HexColor("#c0392b") if status == "已驳回" else colors.HexColor("#1e7a50")

    story.append(Paragraph("低产风险预警报告", TITLE))
    story.append(Paragraph(f"丰谷智警 · PhysCrop-Risk 报告 Agent", META))
    story.append(Spacer(1, 3 * mm))
    meta_rows = [
        ["田块", report["event_key"]],
        ["报告编号", report["report_id"]],
        ["状态", status],
        ["生成方式", GENERATOR_ZH.get(provenance.get("generator"), provenance.get("generator", ""))],
        ["引擎 / 模型", f"{PROVIDER_ZH.get(provenance.get('provider', 'local'), provenance.get('provider', ''))} / {provenance.get('model', '—')}"],
        ["生成时间", (report.get("generated_at") or "").replace("T", " ")[:19]],
        ["决策口径", "修正口径（仅校准参考场）"],
    ]
    meta_table = Table(meta_rows, colWidths=[30 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6a7a70")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f7f5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe6e1")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1c2a22")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"结论：{p['conclusion']}", BODY))
    story.append(Paragraph(p["narrative"], BODY))

    _section(story, "一、农户摘要")
    story.append(Paragraph(report["farmer_summary"]["headline"], _style("headline", 11, 17, True)))
    story.append(Paragraph(report["farmer_summary"]["what_we_observed"], BODY))
    story.append(Paragraph("现在可以做什么", _style("sub", 10, 16, True)))
    _bullets(story, report["farmer_summary"]["what_to_do_now"])
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(report["farmer_summary"]["important_note"], NOTE))

    _section(story, "二、观测轨迹解读")
    story.append(Paragraph(p["trajectory_assessment"]["narrative"], BODY))

    _section(story, "三、空间证据")
    story.append(Paragraph(p["spatial_assessment"]["narrative"], BODY))

    _section(story, "四、锁定证据项")
    evidence_rows = [[Paragraph("项目", CELL_BOLD), Paragraph("数值", CELL_BOLD), Paragraph("解释边界", CELL_BOLD)]]
    for item in p["evidence_items"]:
        evidence_rows.append([
            Paragraph(str(item["title"]), CELL),
            Paragraph(str(item["value"]), CELL),
            Paragraph(str(item["limitation"]), CELL),
        ])
    evidence_table = Table(evidence_rows, colWidths=[40 * mm, 34 * mm, 76 * mm])
    evidence_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe6e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f7f5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(evidence_table)

    _section(story, "五、分阶段处置计划")
    story.append(Paragraph(report["action_plan"]["rationale"], BODY))
    for key, label in (("within_24_hours", "24 小时内"), ("within_24_to_72_hours", "24 至 72 小时"),
                       ("after_confirmation", "现场确认后"), ("do_not_do", "明确不要做")):
        items = report["action_plan"].get(key) or []
        if items:
            story.append(Paragraph(label, _style("sub", 10, 16, True)))
            _bullets(story, items)

    _section(story, "六、复查与升级")
    story.append(Paragraph(f"建议窗口：{report['recheck_plan']['recommended_window']}", BODY))
    story.append(Paragraph(f"降级条件：{report['recheck_plan']['close_condition']}", BODY))
    story.append(Paragraph(f"升级条件：{report['recheck_plan']['escalate_condition']}", BODY))
    story.append(Paragraph(f"升级规则：{report['escalation']['rule']}", BODY))
    story.append(Paragraph("人工批准：必须", BODY))

    _section(story, "七、局限性与来源")
    _bullets(story, report["limitations"])
    story.append(Paragraph("知识来源", _style("sub", 10, 16, True)))
    for source in report["knowledge_citations"]:
        story.append(Paragraph(f"· {source['title']}（{source.get('organization', '')}）", SMALL))

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#c9d4cd")))
    story.append(Paragraph(
        f"证据哈希：{provenance.get('evidence_hash', '')} · 知识库版本：{provenance.get('knowledge_version', '')}",
        SMALL,
    ))
    story.append(Paragraph(
        "本报告在人工批准前不得作为正式处置指令；p 值不是低产概率，预测区间不是保证范围；"
        "空间证据仅用于田块内相对定位，不是病因诊断。",
        NOTE,
    ))
    story.append(Paragraph(
        f"丰谷智警 · 报告状态：{status} · 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        SMALL,
    ))

    doc.build(story)
    return buffer.getvalue()
