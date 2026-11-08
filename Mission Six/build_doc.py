"""Convert mission-06-doc.md into mission-06-doc.pdf with reportlab (same approach as Mission Five)."""
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#153B50")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyM", fontName="Helvetica", fontSize=9.3, leading=12.5, spaceAfter=7))
styles.add(ParagraphStyle(name="CellM", fontName="Helvetica", fontSize=8, leading=10.4))
styles.add(ParagraphStyle(name="CodeM", fontName="Courier", fontSize=6.9, leading=8.6))
styles.add(ParagraphStyle(name="CoverM", parent=styles["Title"], textColor=NAVY, fontSize=19, leading=24,
                          alignment=TA_CENTER, spaceAfter=6))
styles["Heading2"].fontSize = 13
styles["Heading2"].textColor = NAVY
story = []


def inline(s):
    s = escape(s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#126A8D">\1</link>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<i>\1</i>", s)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', s)


def table(rows):
    n = len(rows[0])
    widths = {2: [110, 378], 3: [150, 50, 288], 4: [110, 110, 110, 158], 5: [78, 58, 160, 154, 38],
              7: [102, 50, 58, 52, 46, 112, 68]}.get(n, [488 / n] * n)
    data = [[Paragraph(inline(c), styles["CellM"]) for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DFECF1")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#BCCDD5")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([t, Spacer(1, 8)])


lines = (ROOT / "mission-06-doc.md").read_text().splitlines()
i, paragraph, code, covers = 0, [], False, 0


def flush():
    if paragraph:
        story.append(Paragraph(inline(" ".join(paragraph)), styles["BodyM"]))
        paragraph.clear()


while i < len(lines):
    raw = lines[i]
    line = raw.strip()
    i += 1
    if line.startswith("```"):
        flush()
        code = not code
        story.append(Spacer(1, 3))
        continue
    if code:
        story.append(Paragraph(escape(raw).replace(" ", "&nbsp;") or "&nbsp;", styles["CodeM"]))
        continue
    if not line:
        flush()
        continue
    if line == "<!-- pagebreak -->":
        flush()
        story.append(PageBreak())
        continue
    if line.startswith("|"):
        flush()
        rows = []
        while True:
            row = [c.strip() for c in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in row):
                rows.append(row)
            if i >= len(lines) or not lines[i].strip().startswith("|"):
                break
            line = lines[i].strip()
            i += 1
        table(rows)
        continue
    m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
    if m:
        flush()
        from PIL import Image as PILImage
        path = ROOT / m.group(2)
        with PILImage.open(path) as image:
            w, h = image.size
        story.extend([Image(str(path), width=470, height=470 * h / w), Spacer(1, 6)])
        continue
    if line.startswith("# "):
        flush()
        covers += 1
        story.append(Paragraph(inline(line[2:]), styles["CoverM"]))
        if covers == 2:
            story.append(Spacer(1, 10))
        continue
    if line.startswith("## "):
        flush()
        story.append(Paragraph(inline(line[3:]), styles["Heading2"]))
        continue
    if line.startswith("- "):
        flush()
        story.append(Paragraph("&#8226; " + inline(line[2:]), styles["BodyM"]))
        continue
    if re.match(r"\d+\. ", line):
        flush()
        story.append(Paragraph(inline(line), styles["BodyM"]))
        continue
    paragraph.append(line)
flush()


def footer(c, d):
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#526873"))
    c.drawString(53, 26, "SQUADRON FEI-FEI | MISSION 06")
    c.drawRightString(542, 26, str(d.page))


SimpleDocTemplate(str(ROOT / "mission-06-doc.pdf"), pagesize=(595, 842), leftMargin=53, rightMargin=54,
                  topMargin=42, bottomMargin=45).build(story, onFirstPage=footer, onLaterPages=footer)
print("Built mission-06-doc.pdf from mission-06-doc.md")
