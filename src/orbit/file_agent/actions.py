import shutil
from pathlib import Path

from orbit.file_agent.scope_guard import check_path_allowed


def move_file(source: Path, destination: Path) -> Path:
    """Move `source` to `destination`. Both paths are checked against the
    scope guardrail before anything on disk changes."""
    resolved_source = check_path_allowed(source)
    resolved_destination = check_path_allowed(destination)

    if not resolved_source.is_file():
        raise FileNotFoundError(f"No such file: {resolved_source}")

    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(resolved_source), str(resolved_destination))
    return resolved_destination


def rename_file(source: Path, new_name: str) -> Path:
    """Rename `source` to `new_name` within the same directory. Both the
    original and resulting paths are checked against the scope guardrail."""
    resolved_source = check_path_allowed(source)
    destination = check_path_allowed(resolved_source.with_name(new_name))

    if not resolved_source.is_file():
        raise FileNotFoundError(f"No such file: {resolved_source}")

    resolved_source.rename(destination)
    return destination
