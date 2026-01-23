#!/usr/bin/env python3
"""Build docs/tutorial.pdf from docs/tutorial.md using ReportLab."""
from __future__ import annotations

import datetime as _dt
import re
import textwrap
from pathlib import Path
from typing import List, Tuple

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        ListFlowable,
        ListItem,
        PageBreak,
        PageTemplate,
        Paragraph,
        Preformatted,
        Spacer,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
except Exception as exc:  # pragma: no cover - build-time dependency only
    raise SystemExit(
        "ReportLab is required. Create a venv and install with: pip install reportlab"
    ) from exc

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "tutorial.md"
PDF_PATH = ROOT / "docs" / "tutorial.pdf"

# Ensure lowercase font names used by the parser map to the built-in fonts.
registerFontFamily(
    "courier",
    normal="Courier",
    bold="Courier-Bold",
    italic="Courier-Oblique",
    boldItalic="Courier-BoldOblique",
)
addMapping("courier", 0, 0, "Courier")
addMapping("courier", 1, 0, "Courier-Bold")
addMapping("courier", 0, 1, "Courier-Oblique")
addMapping("courier", 1, 1, "Courier-BoldOblique")


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
NUMBER_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
FENCE_RE = re.compile(r"^```")


class TutorialDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, on_page, **kwargs):
        super().__init__(filename, **kwargs)
        self._bookmark_id = 0
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        template = PageTemplate(id="normal", frames=[frame], onPage=on_page)
        self.addPageTemplates([template])

        self.toc = TableOfContents()
        self.toc.levelStyles = [
            ParagraphStyle(
                name="TOCLevel1",
                fontName="Helvetica",
                fontSize=9,
                leading=11,
                leftIndent=10,
                firstLineIndent=-10,
                spaceBefore=4,
            ),
            ParagraphStyle(
                name="TOCLevel2",
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                leftIndent=24,
                firstLineIndent=-12,
                spaceBefore=2,
            ),
        ]

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and getattr(flowable, "_toc_level", None) is not None:
            level = int(flowable._toc_level)
            text = flowable.getPlainText()
            max_len = 72 if level == 0 else 84
            if len(text) > max_len:
                text = text[: max_len - 3].rstrip() + "..."
            key = f"h{level}-{self._bookmark_id}"
            self._bookmark_id += 1
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))

    def beforeDocument(self):
        self._bookmark_id = 0


def _load_project_metadata() -> Tuple[str, str | None]:
    name = "SOZLab"
    version = None
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        name = project.get("name", name)
        if isinstance(name, str) and name.lower() == "sozlab":
            name = "SOZLab"
        version = project.get("version")
    return name, version


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline_format(text: str) -> str:
    text = _escape(text)
    text = re.sub(r"`([^`]+)`", r"<b>\1</b>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _wrap_code_block(code: str, max_width: float, font_name: str, font_size: float) -> str:
    safe_code = code.replace("\t", "    ")
    char_width = pdfmetrics.stringWidth("M", font_name, font_size) or 6.0
    max_chars = max(20, int(max_width / char_width) - 1)
    wrapped_lines: List[str] = []
    for line in safe_code.splitlines():
        if not line:
            wrapped_lines.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip(" ")
        if indent >= max_chars:
            indent = 0
        width = max(10, max_chars - indent)
        if len(line) <= max_chars:
            wrapped_lines.append(line)
            continue
        for segment in textwrap.wrap(
            stripped,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        ):
            wrapped_lines.append(" " * indent + segment)
    return "\n".join(wrapped_lines)


def _parse_markdown(text: str) -> List[Tuple[str, object]]:
    lines = text.splitlines()
    blocks: List[Tuple[str, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if FENCE_RE.match(stripped):
            code_lines = []
            i += 1
            while i < len(lines) and not FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines) and FENCE_RE.match(lines[i].strip()):
                i += 1
            blocks.append(("code", "\n".join(code_lines)))
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            blocks.append(("heading", (level, title)))
            i += 1
            continue

        bullet = BULLET_RE.match(line)
        number = NUMBER_RE.match(line)
        if bullet or number:
            list_type = "bullet" if bullet else "number"
            items: List[str] = []
            while i < len(lines):
                bullet = BULLET_RE.match(lines[i])
                number = NUMBER_RE.match(lines[i])
                if list_type == "bullet" and bullet:
                    items.append(bullet.group(1).strip())
                elif list_type == "number" and number:
                    items.append(number.group(2).strip())
                else:
                    break
                i += 1
            blocks.append(("list", (list_type, items)))
            continue

        if stripped == "":
            blocks.append(("blank", ""))
            i += 1
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if (
                next_line.strip() == ""
                or HEADING_RE.match(next_line)
                or BULLET_RE.match(next_line)
                or NUMBER_RE.match(next_line)
                or FENCE_RE.match(next_line.strip())
            ):
                break
            para_lines.append(next_line.strip())
            i += 1
        blocks.append(("paragraph", " ".join(para_lines)))

    return blocks


def _build_story(blocks: List[Tuple[str, object]], styles, doc_width: float) -> List[object]:
    story: List[object] = []
    heading_styles = {
        1: styles["H1"],
        2: styles["H2"],
        3: styles["H3"],
        4: styles["H4"],
    }

    for kind, payload in blocks:
        if kind == "heading":
            level, title = payload
            style = heading_styles.get(level, styles["H4"])
            para = Paragraph(_inline_format(title), style)
            if level <= 2:
                para._toc_level = level - 1
            story.append(para)
            continue

        if kind == "paragraph":
            story.append(Paragraph(_inline_format(str(payload)), styles["Body"]))
            continue

        if kind == "list":
            list_type, items = payload
            list_items = [
                ListItem(Paragraph(_inline_format(item), styles["Body"]))
                for item in items
            ]
            bullet_type = "1" if list_type == "number" else "bullet"
            list_flow = ListFlowable(
                list_items,
                bulletType=bullet_type,
                start="1",
                leftIndent=16,
                bulletFontName="Helvetica",
                bulletFontSize=9,
            )
            story.append(list_flow)
            continue

        if kind == "code":
            available = doc_width - styles["Code"].leftIndent - styles["Code"].rightIndent
            wrapped = _wrap_code_block(
                str(payload),
                available,
                styles["Code"].fontName,
                styles["Code"].fontSize,
            )
            story.append(Preformatted(wrapped, styles["Code"]))
            continue

        if kind == "blank":
            story.append(Spacer(1, 6))
            continue

    return story


def build_pdf() -> None:
    if not MD_PATH.exists():
        raise SystemExit(f"Missing source file: {MD_PATH}")

    project_name, version = _load_project_metadata()
    date_str = _dt.date.today().strftime("%B %d, %Y")

    page_width, page_height = LETTER

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(page_width / 2.0, 0.5 * 72, str(doc.page))
        canvas.restoreState()

    doc = TutorialDocTemplate(
        str(PDF_PATH),
        pagesize=LETTER,
        leftMargin=54,
        rightMargin=54,
        topMargin=60,
        bottomMargin=54,
        on_page=_on_page,
    )

    styles = {
        "Title": ParagraphStyle(
            name="Title",
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            name="Subtitle",
            fontName="Helvetica",
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=18,
        ),
        "Meta": ParagraphStyle(
            name="Meta",
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
        ),
        "H1": ParagraphStyle(
            name="H1",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            name="H2",
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            name="H3",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=8,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "H4": ParagraphStyle(
            name="H4",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            name="Body",
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceBefore=2,
            spaceAfter=6,
        ),
        "Code": ParagraphStyle(
            name="Code",
            fontName="Courier",
            fontSize=9,
            leading=11,
            leftIndent=6,
            rightIndent=6,
            spaceBefore=6,
            spaceAfter=6,
            backColor=colors.HexColor("#f2f2f2"),
        ),
    }

    story: List[object] = []

    # Title page
    tagline = "Linux GUI + CLI for solvent occupancy zone analysis"
    story.append(Spacer(1, 120))
    story.append(Paragraph(project_name, styles["Title"]))
    story.append(Paragraph(tagline, styles["Subtitle"]))
    meta_lines = [f"Date: {date_str}"]
    if version:
        meta_lines.insert(0, f"Version: {version}")
    story.append(Paragraph(" | ".join(meta_lines), styles["Meta"]))
    story.append(PageBreak())

    # Table of contents
    story.append(Paragraph("Table of Contents", styles["H1"]))
    story.append(Spacer(1, 12))
    story.append(doc.toc)
    story.append(Spacer(1, 12))

    blocks = _parse_markdown(MD_PATH.read_text(encoding="utf-8"))
    story.extend(_build_story(blocks, styles, doc.width))

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.multiBuild(story, maxPasses=20)

    if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
        raise SystemExit("PDF generation failed: output file is missing or empty")

    size_kb = PDF_PATH.stat().st_size / 1024.0
    print("Tutorial PDF build complete")
    print(f"Source: {MD_PATH}")
    print(f"Output: {PDF_PATH} ({size_kb:.1f} KB)")
    print("Notes: code blocks are wrapped to fit the page width")


if __name__ == "__main__":
    build_pdf()
