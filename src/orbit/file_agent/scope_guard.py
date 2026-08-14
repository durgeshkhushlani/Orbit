from pathlib import Path

from orbit.config import settings


class ScopeViolation(Exception):
    """Raised when a path falls outside the configured ORBIT_ALLOWED_DIRS.

    This is a hard refusal, not something the human Confirm? gate can override --
    it must be checked before any File Agent action is even planned.
    """


def check_path_allowed(path: Path) -> Path:
    """Resolve `path` (following symlinks, rejecting `../` traversal) and verify
    it falls under one of the allowed root directories. Returns the resolved
    path on success; raises ScopeViolation otherwise."""
    resolved = path.resolve()

    for allowed_root in settings.allowed_dirs:
        if resolved == allowed_root or allowed_root in resolved.parents:
            return resolved

    raise ScopeViolation(
        f"'{path}' is outside the folders I'm allowed to touch. "
        "Add it to ORBIT_ALLOWED_DIRS if this should be allowed."
    )
