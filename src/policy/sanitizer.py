import re
from typing import List

# Matches any DML/DDL keyword that must never appear in a user query
_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|EXEC|EXECUTE|CALL|MERGE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class SQLSanitizer:
    """Applies security constraints (column whitelists, row filters) to SQL queries."""

    @staticmethod
    def is_read_only(sql: str) -> bool:
        """Returns True only if the statement is a plain SELECT or a CTE starting with WITH."""
        first_token = sql.strip().lstrip(";").strip().split()[0].upper() if sql.strip() else ""
        if first_token not in ("SELECT", "WITH"):
            return False
        return not bool(_DML_PATTERN.search(sql))

    @staticmethod
    def apply_constraints(sql: str, allowed_columns: List[str], row_filter: str) -> str:
        """Enforce security by pruning columns and injecting WHERE clauses."""
        sanitized_sql = sql.strip().rstrip(";")

        # 1. Apply column restrictions
        if allowed_columns:
            select_match = re.search(
                r"SELECT\s+(.*?)\s+FROM", sanitized_sql, re.IGNORECASE | re.DOTALL
            )
            if select_match:
                original_cols = select_match.group(1).strip()
                requested = [c.strip() for c in original_cols.split(",")]
                if original_cols == "*" or any(c not in allowed_columns for c in requested):
                    sanitized_sql = sanitized_sql.replace(
                        original_cols, ", ".join(allowed_columns), 1
                    )

        # 2. Inject row-level filter before GROUP BY / ORDER BY / LIMIT
        if row_filter:
            insert_pos = len(sanitized_sql)
            for keyword in (" GROUP BY ", " ORDER BY ", " LIMIT "):
                idx = sanitized_sql.upper().find(keyword)
                if idx != -1 and idx < insert_pos:
                    insert_pos = idx

            main_part = sanitized_sql[:insert_pos]
            trailing_part = sanitized_sql[insert_pos:]

            if " WHERE " in main_part.upper():
                sanitized_sql = f"{main_part} AND ({row_filter}){trailing_part}"
            else:
                sanitized_sql = f"{main_part} WHERE {row_filter}{trailing_part}"

        return sanitized_sql.strip()
