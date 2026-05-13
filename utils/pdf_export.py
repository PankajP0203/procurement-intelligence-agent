import html
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

_HEADER_NAVY = colors.HexColor("#1a3a5c")
_RULE_BLUE = colors.HexColor("#2563eb")
_SECTION_BLUE = colors.HexColor("#1e40af")


def _is_rule_line(line: str) -> bool:
    s = line.strip()
    return len(s) > 3 and all(c in "─-═ " for c in s)


def _is_section_header(line: str) -> bool:
    s = line.strip()
    return s.endswith(":") and s == s.upper() and len(s) > 2


def _safe(text: str) -> str:
    return html.escape(text)


def generate_memo_pdf(memo_text: str) -> bytes:
    """Render memo_text as a formatted PDF and return the raw bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MemoTitle",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        textColor=_HEADER_NAVY,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    ts_style = ParagraphStyle(
        "Timestamp",
        parent=base["Normal"],
        fontSize=8.5,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=_SECTION_BLUE,
        spaceBefore=14,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=base["Normal"],
        fontSize=10,
        leading=15,
        leftIndent=14,
        spaceAfter=2,
    )

    story = []

    # Fixed PDF header
    story.append(Paragraph("PROCUREMENT APPROVAL MEMO", title_style))
    ts = datetime.now().strftime("%B %d, %Y at %H:%M")
    story.append(Paragraph(f"Generated {ts}", ts_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_RULE_BLUE, spaceAfter=16))

    # Strip LLM-generated title/rule lines at the top of the text
    lines = memo_text.splitlines()
    start = 0
    if lines and "PROCUREMENT APPROVAL MEMO" in lines[0].upper():
        start = 1
        if len(lines) > 1 and _is_rule_line(lines[1]):
            start = 2

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
        elif _is_section_header(stripped):
            story.append(Paragraph(_safe(stripped), section_style))
        elif stripped.startswith(("- ", "• ")):
            story.append(Paragraph(f"• {_safe(stripped[2:].strip())}", bullet_style))
        else:
            story.append(Paragraph(_safe(stripped), body_style))

    doc.build(story)
    return buf.getvalue()
