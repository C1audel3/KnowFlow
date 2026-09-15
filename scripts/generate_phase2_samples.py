#!/usr/bin/env python
"""Generate deterministic phase-two PDF and PNG fixtures."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportLabImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
IMAGE_PATH = SAMPLE_DIR / "phase2_image.png"
PDF_PATH = SAMPLE_DIR / "phase2_multimodal.pdf"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def generate_image() -> None:
    canvas = Image.new("RGB", (1200, 650), "white")
    draw = ImageDraw.Draw(canvas)
    navy = "#17365D"
    blue = "#DCE6F1"
    green = "#E2F0D9"
    orange = "#FCE4D6"

    draw.rounded_rectangle((20, 20, 1180, 630), radius=24, outline=navy, width=6)
    draw.text((330, 55), "ATLAS MULTIMODAL PIPELINE", fill=navy, font=_font(44))

    boxes = [
        ((70, 220, 320, 400), blue, "DOCUMENTS"),
        ((475, 220, 725, 400), green, "ORION\nVISION GATEWAY"),
        ((880, 220, 1130, 400), orange, "KNOWLEDGE\nGRAPH"),
    ]
    for bounds, color, label in boxes:
        draw.rounded_rectangle(bounds, radius=18, fill=color, outline=navy, width=4)
        lines = label.splitlines()
        for index, line in enumerate(lines):
            left, top, right, _ = bounds
            text_box = draw.textbbox((0, 0), line, font=_font(30))
            width = text_box[2] - text_box[0]
            draw.text(
                ((left + right - width) / 2, top + 58 + index * 45),
                line,
                fill=navy,
                font=_font(30),
            )

    for start, end in [((325, 310), (465, 310)), ((730, 310), (870, 310))]:
        draw.line((start, end), fill=navy, width=8)
        draw.polygon(
            [(end[0], end[1]), (end[0] - 24, end[1] - 14), (end[0] - 24, end[1] + 14)],
            fill=navy,
        )

    draw.text((435, 505), "IMAGE CODE: PIXEL-42", fill=navy, font=_font(34))
    canvas.save(IMAGE_PATH, optimize=True)


def generate_pdf() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Phase2Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
    )
    body = styles["BodyText"]
    body.fontSize = 11
    body.leading = 16

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Atlas Phase 2 Multimodal Report",
        author="KnowFlow",
        invariant=1,
    )
    story = [
        Paragraph("Atlas Phase 2 Multimodal Report", title),
        Spacer(1, 5 * mm),
        Paragraph(
            "Project Atlas targets release in 2027 Q3. The pilot validates real "
            "document parsing, multimodal indexing, and mixed retrieval.",
            body,
        ),
        Spacer(1, 4 * mm),
        ReportLabImage(str(IMAGE_PATH), width=168 * mm, height=91 * mm),
        Paragraph(
            "Figure 1. The ORION vision gateway connects documents to the knowledge graph.",
            body,
        ),
        Spacer(1, 5 * mm),
    ]

    table = Table(
        [
            ["Sensor", "Latency", "Accuracy"],
            ["Sensor-A", "24 ms", "91%"],
            ["Sensor-B", "18 ms", "94%"],
            ["Sensor-C", "31 ms", "89%"],
        ],
        colWidths=[55 * mm, 45 * mm, 45 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#17365D")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
            ]
        )
    )
    story.extend(
        [
            Paragraph("Table 1. Pilot sensor results", styles["Heading3"]),
            table,
            Spacer(1, 5 * mm),
            Paragraph("Evaluation formula", styles["Heading3"]),
            Paragraph(
                "F<sub>1</sub> = 2 x (Precision x Recall) / (Precision + Recall)",
                ParagraphStyle(
                    "Formula",
                    parent=styles["BodyText"],
                    alignment=TA_CENTER,
                    fontName="Times-Italic",
                    fontSize=16,
                    leading=22,
                ),
            ),
            Paragraph(
                "The F1 score is the harmonic mean of precision and recall.", body
            ),
        ]
    )
    doc.build(story)


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    generate_image()
    generate_pdf()
    print(f"generated {IMAGE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"generated {PDF_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
