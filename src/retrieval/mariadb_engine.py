from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


@lru_cache(maxsize=4)
def get_engine(database_url: str, statement_timeout_seconds: int) -> Engine:
    """Shared, process-lifetime-cached engine for a given DATABASE_URL.

    Caching avoids SchemaProvider and DBClient each opening their own
    connection pool. The statement-timeout hook runs on every new physical
    connection so a pathological LLM-generated query can't hang the
    session indefinitely — MariaDB's MAX_STATEMENT_TIME is a session
    variable, not something expressible in the connection string itself.
    """
    engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800)

    @event.listens_for(engine, "connect")
    def _set_statement_timeout(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute(f"SET SESSION MAX_STATEMENT_TIME={statement_timeout_seconds}")
        cursor.close()

    return engine
