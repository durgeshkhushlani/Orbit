from pathlib import Path

from docx import Document
from fpdf import FPDF

from orbit.file_agent.scope_guard import check_path_allowed


def write_markdown(content: str, destination: Path) -> Path:
    """Write `content` to `destination` as a Markdown file. The output path is
    scope-checked for consistency with the other agents, even though document
    generation itself has no Confirm? gate."""
    resolved = check_path_allowed(destination)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return resolved


def write_docx(content: str, destination: Path) -> Path:
    """Write `content` to `destination` as a .docx file, one paragraph per
    blank-line-separated block."""
    resolved = check_path_allowed(destination)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    for block in content.split("\n\n"):
        document.add_paragraph(block.strip())
    document.save(str(resolved))
    return resolved


def write_pdf(content: str, destination: Path) -> Path:
    """Write `content` to `destination` as a simple single-column PDF."""
    resolved = check_path_allowed(destination)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(str(resolved))
    return resolved
