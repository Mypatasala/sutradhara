import re
from typing import List, Optional

class SQLSanitizer:
    """
    Applies security constraints (column whitelists, row filters) to SQL queries.
    """
    
    @staticmethod
    def apply_constraints(sql: str, allowed_columns: List[str], row_filter: str) -> str:
        """
        Enforce security by pruning columns and injecting WHERE clauses.
        """
        sanitized_sql = sql.strip()
        if sanitized_sql.endswith(";"):
            sanitized_sql = sanitized_sql[:-1].strip()
        
        # 1. Apply Column Restrictions (if provided)
        if allowed_columns:
            # Simple regex to find the SELECT part
            # This handles 'SELECT *' or 'SELECT col1, col2'
            # Note: This is an architectural demonstration; a production parser should be used for complex queries.
            select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sanitized_sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                original_cols = select_match.group(1).strip()
                if original_cols == "*" or any(c not in allowed_columns for c in [x.strip() for x in original_cols.split(",")]):
                    # Replace with whitelist
                    sanitizer_cols = ", ".join(allowed_columns)
                    sanitized_sql = sanitized_sql.replace(original_cols, sanitizer_cols, 1)

        # 2. Apply Row Filters (if provided)
        if row_filter:
            # Find the position BEFORE any trailing keywords
            insert_pos = len(sanitized_sql)
            for keyword in [" GROUP BY ", " ORDER BY ", " LIMIT "]:
                idx = sanitized_sql.upper().find(keyword)
                if idx != -1 and idx < insert_pos:
                    insert_pos = idx
            
            main_part = sanitized_sql[:insert_pos]
            trailing_part = sanitized_sql[insert_pos:]

            if " WHERE " in main_part.upper():
                sanitized_sql = f"{main_part} AND ({row_filter}) {trailing_part}"
            else:
                sanitized_sql = f"{main_part} WHERE {row_filter} {trailing_part}"

        return sanitized_sql.strip()
