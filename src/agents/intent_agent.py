import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..retrieval.schema_provider import SchemaProvider

class IntentResolutionAgent:
    """
    A database-aware SQL generation agent.
    Analyzes schema at runtime and generates SQL queries strictly based on the provided schema.
    """
    def __init__(self):
        self.schema_provider = SchemaProvider()
        self.models = self._setup_models()

    def _setup_models(self):
        """Setup available LLMs based on environment variables."""
        models = []
        # Support both Gemini and OpenAI
        if os.getenv("OPENAI_API_KEY"):
            models.append(ChatOpenAI(model="gpt-4o-mini"))
        if os.getenv("GOOGLE_API_KEY"):
            models.append(ChatGoogleGenerativeAI(model="gemini-flash-latest"))
        return models

    async def resolve(self, query: str, schema_summary: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parses natural language query into SQL using context and schema.
        """
        if not self.models:
            return {"error": "No LLM API keys configured. Please set GOOGLE_API_KEY or OPENAI_API_KEY."}

        from datetime import datetime
        context_str = f"User Context: {context}\nToday's Date: {datetime.now().strftime('%Y-%m-%d')}" if context else f"Today's Date: {datetime.now().strftime('%Y-%m-%d')}"
        
        system_prompt = f"""
You are a database-aware SQL generation agent.
{context_str}

Your task is to analyze the provided database schema at runtime and generate a correct, executable SQL query based strictly on:
- The user’s natural language question
- The dynamically supplied schema (tables, columns, data types, relationships)

CORE RULES:
1. Do NOT hardcode any table names, column names, or conditions.
2. Use ONLY the tables and columns present in the provided schema.
3. If a required field is not present in the schema, return a clarification question starting with "CLARIFICATION: ".
4. Infer joins only from foreign key relationships or matching column semantics in the schema.
5. Generate SQL that is: Syntactically valid, Optimized, Read-only.
6. Do not assume WHERE clauses unless explicitly implied.
7. Never hallucinate columns, tables, or filters.
8. Do NOT include a semicolon at the end of the SQL query.
9. ALWAYS return your response in the following structured format:
   TABLE: <primary_table_name>
   ACTION: <select|aggregate|insert|update|delete>
   SQL: <the_generated_sql_query>

{schema_summary}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        last_exception = None
        for model in self.models:
            try:
                response = await model.ainvoke(messages)
                if isinstance(response.content, list):
                    content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
                else:
                    content = str(response.content).strip()

                # Clean markdown code blocks if present
                if "```" in content:
                    # Extracts content between backticks or just trims if it's the whole thing
                    content = re.sub(r"```(sql)?", "", content).strip()

                if content.startswith("CLARIFICATION:"):
                    return {
                        "clarification": {
                            "question": content.replace("CLARIFICATION:", "").strip(),
                            "options": [] # Defaulting to empty options for simple prompts
                        }
                    }

                # Parse the structured response
                result = {"table": "unknown", "action": "select", "sql": ""}
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("TABLE:"):
                        result["table"] = line.replace("TABLE:", "").strip()
                    elif line.startswith("ACTION:"):
                        result["action"] = line.replace("ACTION:", "").strip().lower()
                    elif line.startswith("SQL:"):
                        result["sql"] = content.split("SQL:")[1].strip()
                
                # Fallback if parsing failed
                if not result["sql"]:
                   result["sql"] = content

                return result
            except Exception as e:
                print(f"LLM Model Error ({model.__class__.__name__}): {str(e)}")
                last_exception = e
                continue
        
        return {"error": f"Internal Model Error: {str(last_exception)}"}
