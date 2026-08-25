import re
import sqlite3
import os
from typing import List, Dict, Any

from sqlalchemy import text

from ..policy.sanitizer import SQLSanitizer
from .mariadb_engine import get_engine

# Matches an unresolved template token like <your_student_id> or {value} — a
# small local LLM occasionally emits one of these in place of a real literal
# it doesn't have (usually an identity/tenant filter it was told to omit
# instead — see intent_agent.py rule 17), which is not valid SQL and would
# otherwise surface as an opaque MariaDB syntax error several layers down.
_PLACEHOLDER_PATTERN = re.compile(r"<[a-zA-Z_][\w ]*>|\{[a-zA-Z_]\w*\}")


class DBClient:
    """Executes read-only SQL against the configured database.

    Backed by SQLite (DB_PATH) by default; set DATABASE_URL to a SQLAlchemy
    URL to read from MariaDB instead — used for the myPatasala integration.
    That connection should always be a read-only DB user (see
    docs/architecture/Sutradhara-Ask-Integration-Status.md) — SQLSanitizer
    below is an app-level backstop, not a substitute for the DB-level grant.
    """

    def __init__(self):
        self.db_path = os.getenv("DB_PATH", "school.db")
        self.database_url = os.getenv("DATABASE_URL")
        self.max_results = int(os.getenv("MAX_QUERY_RESULTS", "1000"))
        self.statement_timeout_seconds = int(os.getenv("DB_STATEMENT_TIMEOUT_SECONDS", "5"))

    def execute(self, sql: str) -> List[Dict[str, Any]]:
        # Block any non-SELECT statements before touching the database
        if not SQLSanitizer.is_read_only(sql):
            return [{"error": "Only SELECT queries are permitted."}]

        if _PLACEHOLDER_PATTERN.search(sql):
            return [{"error": "Generated SQL contains an unresolved placeholder instead of a real value."}]

        if self.database_url:
            return self._execute_sqlalchemy(sql)
        return self._execute_sqlite(sql)

    def _execute_sqlite(self, sql: str) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return [{"error": f"Database '{self.db_path}' not found."}]

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(self.max_results)
            result = [dict(row) for row in rows]
            conn.close()
            return result
        except Exception as e:
            return [{"error": str(e)}]

    def _execute_sqlalchemy(self, sql: str) -> List[Dict[str, Any]]:
        try:
            engine = get_engine(self.database_url, self.statement_timeout_seconds)
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchmany(self.max_results)
                return [dict(row._mapping) for row in rows]
        except Exception as e:
            return [{"error": str(e)}]
