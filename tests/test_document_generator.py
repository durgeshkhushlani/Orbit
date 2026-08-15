import pytest
from docx import Document

from orbit.config import settings
from orbit.document_agent.generator import write_docx, write_markdown, write_pdf
from orbit.file_agent.scope_guard import ScopeViolation


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def test_write_markdown_creates_file_with_content(allowed_root):
    destination = allowed_root / "notes.md"

    result = write_markdown("# Title\n\nSome content.", destination)

    assert result == destination.resolve()
    assert result.read_text(encoding="utf-8") == "# Title\n\nSome content."


def test_write_docx_creates_readable_document(allowed_root):
    destination = allowed_root / "report.docx"

    result = write_docx("First paragraph.\n\nSecond paragraph.", destination)

    document = Document(str(result))
    paragraphs = [p.text for p in document.paragraphs]
    assert paragraphs == ["First paragraph.", "Second paragraph."]


def test_write_pdf_creates_nonempty_pdf_file(allowed_root):
    destination = allowed_root / "summary.pdf"

    result = write_pdf("Line one.\nLine two.", destination)

    assert result.exists()
    assert result.read_bytes().startswith(b"%PDF")


def test_write_outside_allowed_dir_is_refused(allowed_root, tmp_path):
    outside = tmp_path / "elsewhere" / "notes.md"

    with pytest.raises(ScopeViolation):
        write_markdown("content", outside)

    assert not outside.exists()
