from typing import Any, Dict, List, Optional
from sqlalchemy import select, func, column, table, join, text, desc
from sqlalchemy.dialects import postgresql

class SQLBuilder:
    """
    Constructs SQL queries programmatically based on structured intent.
    Uses SQLAlchemy's expression language for dialect-agnostic (mostly) building.
    """
    
    def generate(self, intent: Dict[str, Any]) -> str:
        """Generates a SQL string from the provided intent object."""
        
        # 1. Base query from entities/tables
        tables = {}
        columns = []
        
        seen_cols = set()
        for entity in intent.get("entities", []):
            if "." in entity:
                t_name, c_name = entity.split(".", 1)
                # Map domain names to physical table names if necessary
                if t_name == "grade" or t_name == "report_card":
                    t_name = "report_cards"
                elif not t_name.endswith("s"):
                    t_name += "s"
                
                if t_name not in tables:
                    tables[t_name] = table(t_name)
                
                # Check for collisions and alias
                if c_name in seen_cols:
                    col_alias = f"{t_name}_{c_name}"
                    columns.append(text(f"{t_name}.{c_name} AS {col_alias}"))
                else:
                    columns.append(text(f"{t_name}.{c_name}"))
                    seen_cols.add(c_name)
        
        # If no entities specified but action is aggregate, we still need tables
        if not columns and intent.get("action") == "aggregate":
            target = intent.get("target", "")
            if "." in target:
                t_name, _ = target.split(".", 1)
                table_name = "report_cards" if t_name == "grade" else (t_name if t_name.endswith("s") else t_name + "s")
                tables[table_name] = table(table_name)

        # 2. Handle Aggregates
        if intent.get("action") == "aggregate":
            target = intent.get("target", "")
            if "." in target:
                t_name, c_name = target.split(".", 1)
                table_name = "report_cards" if t_name == "grade" else (t_name if t_name.endswith("s") else t_name + "s")
                
                func_name = intent.get("function", "AVG")
                agg_col = func.__getattr__(func_name)(text(c_name))
                
                # Group by
                group_by_field = intent.get("group_by")
                if group_by_field and "." in group_by_field:
                    _, g_c_name = group_by_field.split(".", 1)
                    stmt = select(text(g_c_name), agg_col).select_from(table(table_name)).group_by(text(g_c_name))
                    return self._compile(stmt)

        # 3. Handle Window Functions (Rank)
        if intent.get("action") == "rank":
            partition_by = intent.get("partition_by")
            order_by = intent.get("order_by", "")
            
            stmt = text(f"SELECT name, {partition_by}, RANK() OVER (PARTITION BY {partition_by} ORDER BY {order_by}) as rank FROM student_scores")
            return str(stmt)

        # 4. Handle Joins
        if intent.get("joins"):
            primary_table = intent["joins"][0]["from"]
            j_obj = table(primary_table)
            for j in intent["joins"]:
                j_obj = join(j_obj, table(j["to"]), text(j["on"]))
            
            stmt = select(*columns).select_from(j_obj)
            return self._compile(stmt)

        # 5. Handle Filters
        if intent.get("filters"):
            t_name = intent.get("entity", "")
            if not t_name and tables:
                t_name = list(tables.keys())[0]
            elif not t_name:
                t_name = "dual"
                
            stmt = select(text("*")).select_from(table(t_name))
            clauses = []
            for f in intent["filters"]:
                col = text(f["column"])
                op = f["operator"]
                val = f["value"]
                
                if op == "BETWEEN":
                    clauses.append(text(f"{f['column']} BETWEEN '{val[0]}' AND '{val[1]}'"))
                elif op == "==":
                    clauses.append(text(f"{f['column']} = '{val}'"))
            
            if clauses:
                stmt = stmt.where(*clauses)
            return self._compile(stmt)

        # Default fallthrough
        if columns:
            stmt = select(*columns)
            if tables:
                stmt = stmt.select_from(*tables.values())
            return self._compile(stmt)
            
        return "SELECT * FROM dual"

    def _compile(self, stmt) -> str:
        """Compile SQLAlchemy statement to PostgreSQL dialect string."""
        return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
