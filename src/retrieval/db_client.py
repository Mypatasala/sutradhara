import sqlite3
import os
from typing import List, Dict, Any

class DBClient:
    """
    Handles execution of SQL queries against the local SQLite database.
    """
    def __init__(self, db_path: str = "school.db"):
        self.db_path = db_path

    def execute(self, sql: str) -> List[Dict[str, Any]]:
        """Executes a SQL query and returns results as a list of dictionaries."""
        if not os.path.exists(self.db_path):
            return [{"error": f"Database {self.db_path} not found."}]
            
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row # Enable dict-like access
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Convert sqlite3.Row objects to real dictionaries
            result = [dict(row) for row in rows]
            conn.close()
            return result
        except Exception as e:
            return [{"error": str(e)}]
