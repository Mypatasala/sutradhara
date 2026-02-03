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

    async def resolve(self, query: str, schema_summary: str) -> Dict[str, Any]:
        """
        Parses natural language query into SQL using available LLMs and provided schema.
        """
        if not self.models:
            return {"error": "No LLM API keys configured. Please set GOOGLE_API_KEY or OPENAI_API_KEY."}

        system_prompt = f"""
You are a database-aware SQL generation agent.

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
8. ALWAYS return the pure SQL string as the output, or a clarification question.

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
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip().endswith("```"):
                        lines = lines[:-1]
                    content = "\n".join(lines).strip()
                    if content.lower().startswith("sql"):
                        content = content[3:].strip()

                if content.startswith("CLARIFICATION:"):
                    return {"clarification": content.replace("CLARIFICATION:", "").strip()}

                return {"sql": content}
            except Exception as e:
                print(f"LLM Model Error ({model.__class__.__name__}): {str(e)}")
                last_exception = e
                continue
        
        return {"error": f"Internal Model Error: {str(last_exception)}"}
