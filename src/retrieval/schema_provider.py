import sqlite3
from typing import List, Dict, Any

class SchemaProvider:
    """
    Dynamically extracts schema information from a SQLite database.
    Provides table names, column details, and foreign key relationships.
    """
    def __init__(self, db_path: str = "school.db"):
        self.db_path = db_path

    def get_schema_summary(self) -> str:
        """Returns a string representation of the schema for LLM consumption, including sample values."""
        schema_info = self.get_full_schema()
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
                
                # Fetch distinct sample values
                try:
                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 3;")
                    samples = [row[0] for row in cursor.fetchall()]
                except Exception:
                    samples = []
                
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

if __name__ == "__main__":
    provider = SchemaProvider()
    print(provider.get_schema_summary())
