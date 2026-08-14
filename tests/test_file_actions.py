import pytest

from orbit.config import settings
from orbit.file_agent.actions import move_file, rename_file
from orbit.file_agent.scope_guard import ScopeViolation


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def test_move_file_within_allowed_dir_succeeds(allowed_root):
    source = allowed_root / "invoice.pdf"
    source.write_text("content")
    destination = allowed_root / "archive" / "invoice.pdf"

    result = move_file(source, destination)

    assert result == destination.resolve()
    assert destination.exists()
    assert not source.exists()


def test_move_file_to_outside_destination_is_refused(allowed_root, tmp_path):
    source = allowed_root / "invoice.pdf"
    source.write_text("content")
    outside_destination = tmp_path / "elsewhere" / "invoice.pdf"

    with pytest.raises(ScopeViolation):
        move_file(source, outside_destination)

    assert source.exists()


def test_rename_file_within_allowed_dir_succeeds(allowed_root):
    source = allowed_root / "invoice_draft.pdf"
    source.write_text("content")

    result = rename_file(source, "invoice_final.pdf")

    assert result == allowed_root / "invoice_final.pdf"
    assert result.exists()
    assert not source.exists()


def test_rename_file_outside_allowed_dir_is_refused(tmp_path):
    source = tmp_path / "invoice.pdf"
    source.write_text("content")

    with pytest.raises(ScopeViolation):
        rename_file(source, "renamed.pdf")

    assert source.exists()
