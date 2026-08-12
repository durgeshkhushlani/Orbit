from contextlib import AbstractContextManager

from langgraph.checkpoint.sqlite import SqliteSaver

from orbit.config import settings


def get_checkpointer() -> AbstractContextManager[SqliteSaver]:
    """Return a context manager yielding the SQLite checkpointer used to persist
    and resume graph state across turns, keyed by thread_id."""
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(settings.checkpoint_db_path))
