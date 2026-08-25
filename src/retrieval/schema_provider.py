import sqlite3
import os
from typing import List, Dict, Any, Optional, Set

from sqlalchemy import inspect, text

from .mariadb_engine import get_engine

# Column names never sampled regardless of table — a schema-inspection-time
# backstop against known-sensitive fields leaking into the LLM prompt (see
# docs/architecture/Sutradhara-Ask-Integration-Status.md, "Schema exposure
# scoping"). Matched by bare column name, not table-qualified, since the
# same sensitive concept (e.g. "notes") recurs across unrelated tables.
SENSITIVE_COLUMNS: Set[str] = {
    "password", "allergies", "special_needs", "blood_group",
    "teacher_notes", "principal_notes",
}


class SchemaProvider:
    """
    Dynamically extracts schema information for LLM consumption. Backed by
    SQLite (DB_PATH) by default; set DATABASE_URL to a SQLAlchemy URL
    (e.g. mysql+pymysql://user:pass@host:3306/db) to read from MariaDB
    instead — used for the myPatasala integration.
    """

    def __init__(self):
        self.db_path = os.getenv("DB_PATH", "school.db")
        self.database_url = os.getenv("DATABASE_URL")
        self.statement_timeout_seconds = int(os.getenv("DB_STATEMENT_TIMEOUT_SECONDS", "5"))

    def get_schema_summary(self, covered_tables: Optional[Set[str]] = None) -> str:
        """Returns a string representation of the schema for LLM consumption, including sample values.

        covered_tables, when given, scopes the summary to just those tables
        — used to keep the prompt small and to avoid handing the LLM tables
        a tenant's policies could never authorize it to query anyway.
        """
        schema_info = self.get_full_schema()
        if covered_tables:
            schema_info = {t: d for t, d in schema_info.items() if t in covered_tables}

        summary = "DATABASE SCHEMA (with sample values):\n"

        for table_name, details in schema_info.items():
            col_parts = []
            for c in details['columns']:
                samples = ", ".join([str(s) for s in c['samples'] if s is not None])
                sample_str = f" [samples: {samples}]" if samples else ""
                col_parts.append(f"{c['name']} ({c['type']}){sample_str}")

            cols = "\n  - ".join(col_parts)
            summary += f"- Table '{table_name}':\n  - {cols}\n"
            if details['foreign_keys']:
                for fk in details['foreign_keys']:
                    summary += f"  - FK: {fk['from']} -> {fk['table']}.{fk['to']}\n"

        return summary

    def get_full_schema(self) -> Dict[str, Any]:
        if self.database_url:
            return self._get_full_schema_sqlalchemy()
        return self._get_full_schema_sqlite()

    def _get_full_schema_sqlite(self) -> Dict[str, Any]:
        """Fetches full schema metadata from SQLite, including sample data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            # Columns
            cursor.execute(f"PRAGMA table_info({table});")
            cols_info = cursor.fetchall()

            cols = []
            for r in cols_info:
                col_name = r[1]
                col_type = r[2]

                samples = self._sample_values_sqlite(cursor, table, col_name)

                cols.append({
                    "name": col_name,
                    "type": col_type,
                    "samples": samples
                })

            # Foreign Keys
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            fks = [{"table": r[2], "from": r[3], "to": r[4]} for r in cursor.fetchall()]

            schema[table] = {
                "columns": cols,
                "foreign_keys": fks
            }

        conn.close()
        return schema

    def _sample_values_sqlite(self, cursor, table: str, col_name: str) -> List[Any]:
        if col_name.lower() in SENSITIVE_COLUMNS:
            return []
        try:
            cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 3;")
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def _get_full_schema_sqlalchemy(self) -> Dict[str, Any]:
        """Fetches full schema metadata via SQLAlchemy reflection (MariaDB)."""
        engine = get_engine(self.database_url, self.statement_timeout_seconds)
        inspector = inspect(engine)

        schema: Dict[str, Any] = {}
        with engine.connect() as conn:
            for table in inspector.get_table_names():
                cols = []
                for col in inspector.get_columns(table):
                    # str(col["type"]) collapses a MySQL/MariaDB ENUM down to the
                    # bare word "ENUM", discarding its value list — exactly the
                    # thing a query like "how many are still assigned" needs to
                    # map "assigned" to the right WHERE value. SQLAlchemy keeps
                    # the real values on the type object's `.enums`; a handful of
                    # sample rows aren't a substitute since LIMIT 3 has no
                    # guarantee of covering the column's full value domain.
                    col_type = str(col["type"])
                    enum_values = getattr(col["type"], "enums", None)
                    if enum_values:
                        col_type = f"ENUM({', '.join(enum_values)})"
                    cols.append({
                        "name": col["name"],
                        "type": col_type,
                        "samples": self._sample_values_sqlalchemy(conn, table, col["name"]),
                    })

                fks = []
                for fk in inspector.get_foreign_keys(table):
                    constrained = fk.get("constrained_columns") or []
                    referred = fk.get("referred_columns") or []
                    if constrained and referred:
                        fks.append({
                            "table": fk["referred_table"],
                            "from": constrained[0],
                            "to": referred[0],
                        })

                schema[table] = {"columns": cols, "foreign_keys": fks}

        return schema

    def _sample_values_sqlalchemy(self, conn, table: str, col_name: str) -> List[Any]:
        if col_name.lower() in SENSITIVE_COLUMNS:
            return []
        try:
            result = conn.execute(
                text(f"SELECT DISTINCT `{col_name}` FROM `{table}` WHERE `{col_name}` IS NOT NULL LIMIT 3")
            )
            return [row[0] for row in result.fetchall()]
        except Exception:
            return []


if __name__ == "__main__":
    provider = SchemaProvider()
    print(provider.get_schema_summary())
