"""Render submission Markdown reports to readable Korean PDFs with ReportLab."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FONT_REGULAR = "Malgun"
FONT_BOLD = "Malgun-Bold"


def register_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD
    candidates = [
        (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")),
        (
            Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
            Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont(FONT_REGULAR, regular))
            pdfmetrics.registerFont(TTFont(FONT_BOLD, bold))
            return

    # ReportLab provides Korean CID fonts even when the host has no Korean TTF.
    FONT_REGULAR = "HYSMyeongJo-Medium"
    FONT_BOLD = "HYGoThic-Medium"
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_REGULAR))
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_BOLD))


def clean_inline(text: str) -> str:
    text = text.replace("<br>", "<br/>")
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#315d78">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", rf'<font name="{FONT_REGULAR}">\1</font>', text)
    return text.replace("&", "&amp;").replace("&amp;lt;", "&lt;").replace("&amp;gt;", "&gt;")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleKR", parent=base["Title"], fontName=FONT_BOLD, fontSize=24, leading=32, textColor=colors.HexColor("#18324a"), spaceAfter=15),
        "h1": ParagraphStyle("H1KR", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=17, leading=23, textColor=colors.HexColor("#18324a"), spaceBefore=13, spaceAfter=8),
        "h2": ParagraphStyle("H2KR", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=13, leading=19, textColor=colors.HexColor("#315d78"), spaceBefore=10, spaceAfter=6),
        "h3": ParagraphStyle("H3KR", parent=base["Heading3"], fontName=FONT_BOLD, fontSize=11, leading=16, textColor=colors.HexColor("#e4573d"), spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("BodyKR", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=9.3, leading=15, textColor=colors.HexColor("#263442"), spaceAfter=6),
        "bullet": ParagraphStyle("BulletKR", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=9, leading=14, leftIndent=12, firstLineIndent=-8, bulletIndent=3, spaceAfter=3),
        "quote": ParagraphStyle("QuoteKR", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=9, leading=14, leftIndent=12, rightIndent=12, borderColor=colors.HexColor("#e8a23a"), borderWidth=1, borderPadding=7, backColor=colors.HexColor("#fff9ec"), spaceAfter=8),
        "code": ParagraphStyle("CodeKR", parent=base["Code"], fontName="Courier", fontSize=7.5, leading=10, leftIndent=8, backColor=colors.HexColor("#f4f6f8"), borderPadding=6, spaceAfter=6),
        "caption": ParagraphStyle("CaptionKR", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=7.5, leading=11, textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=7),
        "table": ParagraphStyle("TableKR", parent=base["BodyText"], fontName=FONT_REGULAR, fontSize=6.8, leading=9, alignment=TA_LEFT),
        "table_header": ParagraphStyle("TableHeaderKR", parent=base["BodyText"], fontName=FONT_BOLD, fontSize=6.8, leading=9, textColor=colors.white, alignment=TA_CENTER),
    }


def parse_table(lines: list[str], sty: dict) -> Table:
    rows = []
    for line in lines:
        values = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r"[-: ]+", value or "-") for value in values):
            continue
        style_name = "table_header" if not rows else "table"
        rows.append([Paragraph(clean_inline(value), sty[style_name]) for value in values])
    width = A4[0] - 32 * mm
    col_widths = [width / len(rows[0])] * len(rows[0])
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315d78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d8e0e7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(16 * mm, 9 * mm, "PlaylistPro 제출용 결과서")
    canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"{doc.page}")
    canvas.setStrokeColor(colors.HexColor("#d8e0e7"))
    canvas.line(16 * mm, 13 * mm, A4[0] - 16 * mm, 13 * mm)
    canvas.restoreState()


def markdown_to_story(source: Path):
    sty = styles()
    lines = source.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    in_code = False
    code_lines: list[str] = []
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("```"):
            if in_code:
                story.append(Paragraph("<br/>".join(code_lines), sty["code"]))
                code_lines = []
            in_code = not in_code
            index += 1
            continue
        if in_code:
            code_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            index += 1
            continue
        if not line:
            story.append(Spacer(1, 2.5 * mm))
            index += 1
            continue
        if line.startswith("| "):
            table_lines = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.extend([parse_table(table_lines, sty), Spacer(1, 3 * mm)])
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            image_path = (source.parent / image_match.group(2)).resolve()
            if image_path.exists():
                img = Image(str(image_path))
                max_width = A4[0] - 36 * mm
                max_height = 108 * mm
                scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                story.append(KeepTogether([img, Paragraph(image_match.group(1), sty["caption"])]))
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), sty["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), sty["h1"]))
        elif line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:]), sty["h2"]))
        elif line.startswith("#### "):
            story.append(Paragraph(clean_inline(line[5:]), sty["h3"]))
        elif line.startswith("> "):
            story.append(Paragraph(clean_inline(line[2:]), sty["quote"]))
        elif re.match(r"^[-*] ", line):
            story.append(Paragraph("• " + clean_inline(line[2:]), sty["bullet"]))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(clean_inline(line), sty["bullet"]))
        else:
            story.append(Paragraph(clean_inline(line), sty["body"]))
        index += 1
    return story


def render(source_name: str, output_name: str) -> None:
    source = REPORTS / source_name
    output = REPORTS / output_name
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=18 * mm,
        title=source.stem, author="PlaylistPro Team3",
    )
    document.build(markdown_to_story(source), onFirstPage=header_footer, onLaterPages=header_footer)
    print(output)


if __name__ == "__main__":
    register_fonts()
    render("preprocessing_report.md", "preprocessing_report.pdf")
    render("training_report.md", "training_report.pdf")
