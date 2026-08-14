import pytest

from orbit.config import settings
from orbit.file_agent.scope_guard import ScopeViolation, check_path_allowed


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def test_path_inside_allowed_dir_is_permitted(allowed_root):
    target = allowed_root / "invoices" / "jan.pdf"
    target.parent.mkdir()
    target.touch()

    resolved = check_path_allowed(target)

    assert resolved == target.resolve()


def test_path_outside_allowed_dirs_is_refused(allowed_root, tmp_path):
    outside = tmp_path / "elsewhere" / "secret.pdf"
    outside.parent.mkdir()
    outside.touch()

    with pytest.raises(ScopeViolation):
        check_path_allowed(outside)


def test_traversal_out_of_allowed_dir_is_refused(allowed_root, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    traversal_path = allowed_root / ".." / "elsewhere"

    with pytest.raises(ScopeViolation):
        check_path_allowed(traversal_path)


def test_no_allowed_dirs_configured_refuses_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "orbit_allowed_dirs", "")

    with pytest.raises(ScopeViolation):
        check_path_allowed(tmp_path / "anything.txt")
