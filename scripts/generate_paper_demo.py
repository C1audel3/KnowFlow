"""Create an original synthetic paper PDF for the reading workflow (no API calls)."""

from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image as PDFImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]


def generate(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = output_dir / "demo_pipeline.png"
    image = Image.new("RGB", (1000, 220), "white")
    draw = ImageDraw.Draw(image)
    labels = ("Question", "Retrieve evidence", "Read text + figure", "Cited answer")
    for index, label in enumerate(labels):
        x = 15 + index * 250
        draw.rounded_rectangle(
            (x, 65, x + 220, 155), radius=12, fill="#edf4ff", outline="#315aa5", width=3
        )
        draw.text((x + 10, 98), label, fill="#17315a", font_size=19)
        if index < 3:
            draw.line((x + 222, 110, x + 246, 110), fill="#315aa5", width=3)
            draw.polygon(
                [(x + 246, 110), (x + 238, 104), (x + 238, 116)], fill="#315aa5"
            )
    image.save(figure)
    styles = getSampleStyleSheet()
    story = []

    def heading(text):
        story.append(Paragraph(text, styles["Heading1"]))

    def paragraph(text):
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 12))

    heading("Evidence-Grounded Paper Reading: A Synthetic Demonstration")
    paragraph(
        "Original demo document for KnowFlow Paper. This is not a published research paper. "
        "All experimental numbers below are fabricated for software testing, not measured results."
    )
    heading("Abstract")
    paragraph(
        "We describe a small paper-reading pipeline that retrieves source evidence before generating an answer. "
        "Its goal is to connect explanations with text, figures and tables in a selected document."
    )
    heading("1. Introduction")
    paragraph(
        "Readers need both visual information and the surrounding experimental conditions. "
        "A citation should identify a concrete evidence block and its PDF page."
    )
    story.append(PageBreak())
    heading("2. Method")
    paragraph(
        "Figure 1 presents four stages: receive a question, retrieve evidence, jointly read text and a figure, "
        "and generate a cited answer. Retrieval is restricted to the selected paper."
    )
    story.append(PDFImage(str(figure), width=480, height=106))
    paragraph(
        "Figure 1. Evidence-grounded reading pipeline. The third stage receives both original images and text."
    )
    heading("2.1 Objective")
    paragraph(
        "Equation 1 defines a toy objective: L = L_answer + lambda * L_citation. "
        "L_answer denotes answer error; L_citation denotes citation error; lambda balances the two terms. "
        "This objective is illustrative and is not used to train the application."
    )
    story.append(PageBreak())
    heading("3. Experimental Setup")
    paragraph(
        "Table 1 uses a fictional set of 20 questions. Both fictional systems use the same questions and scoring. "
        "Accuracy is the fraction of correct answers. These values are only test fixtures."
    )
    table = Table(
        [
            ["System", "Correct / Total", "Accuracy"],
            ["Text-only", "14 / 20", "70%"],
            ["Text + original figure", "17 / 20", "85%"],
        ],
        colWidths=[230, 130, 100],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(table)
    paragraph("Table 1. Fictional comparison on the same synthetic questions.")
    heading("4. Results")
    paragraph(
        "The fabricated figures differ by 15 percentage points. They cannot support any claim about actual model quality."
    )
    heading("5. Limitations")
    paragraph(
        "No real users, published benchmark or repeated experiments were evaluated. "
        "A demonstration document does not establish generalization or scientific validity."
    )
    heading("6. Conclusion")
    paragraph(
        "The document demonstrates source-grounded reading interactions, not research effectiveness."
    )

    def page_number(canvas, document):
        canvas.drawRightString(560, 25, str(document.page))

    target = output_dir / "demo_paper.pdf"
    SimpleDocTemplate(str(target)).build(
        story, onFirstPage=page_number, onLaterPages=page_number
    )
    return target


if __name__ == "__main__":
    print(generate(ROOT / "data" / "samples" / "papers"))
