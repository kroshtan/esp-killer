from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url

from server.db.tables import metadata


def make_engine(database_url: str) -> Engine:
    """
    Create an engine and make sure the schema exists.

    For SQLite this also creates the database directory and switches on WAL, so the API and the worker can use
    the same file concurrently (one writer at a time, readers never blocked).

    :param database_url: SQLAlchemy URL, e.g. ``sqlite:///data/espk.db``
    :return: the engine
    """
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    else:
        engine = create_engine(url, pool_pre_ping=True)
    # A schema-version table and real migrations (alembic) come before the first schema change in production.
    metadata.create_all(engine)
    return engine
