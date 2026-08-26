import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..retrieval.schema_provider import SchemaProvider

logger = logging.getLogger(__name__)


class IntentResolutionAgent:
    """
    A database-aware SQL generation agent.
    Analyzes schema at runtime and generates SQL queries strictly based on the provided schema.
    """

    def __init__(self):
        self.schema_provider = SchemaProvider()
        self.models = self._setup_models()

    def _setup_models(self) -> list:
        """Build the ordered list of LLM clients from env config.

        LLM_PROVIDER controls which back-end(s) are used:
          openai  – OpenAI only (requires OPENAI_API_KEY)
          google  – Google Gemini only (requires GOOGLE_API_KEY)
          ollama  – Local Ollama only (requires OLLAMA_BASE_URL or defaults to localhost:11434)
          auto    – All configured providers tried in order: OpenAI → Google → Ollama

        LLM_MODEL overrides the model name for the active provider(s) when set.
        Use provider-specific vars for independent control in auto mode:
          OPENAI_MODEL  (default: gpt-4o-mini)
          GOOGLE_MODEL  (default: gemini-flash-latest)
          OLLAMA_MODEL  (default: llama3.2)
        """
        provider = os.getenv("LLM_PROVIDER", "auto").lower()
        model_override = os.getenv("LLM_MODEL", "")
        models: list = []

        # temperature=0 everywhere: this agent's job is strict structured
        # extraction (TABLE:/ACTION:/SQL: or a clarification), not creative
        # generation — sampling variance here just means the same question
        # sometimes parses and sometimes doesn't. Matters most for smaller
        # local models (e.g. Ollama), which are already less reliable at
        # following an exact output format.

        # --- OpenAI ---
        if provider in ("openai", "auto") and os.getenv("OPENAI_API_KEY"):
            name = model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            models.append(ChatOpenAI(model=name, temperature=0))
            logger.info("LLM: OpenAI/%s", name)

        # --- Google Gemini ---
        if provider in ("google", "auto") and os.getenv("GOOGLE_API_KEY"):
            name = model_override or os.getenv("GOOGLE_MODEL", "gemini-flash-latest")
            models.append(ChatGoogleGenerativeAI(model=name, temperature=0))
            logger.info("LLM: Google/%s", name)

        # --- Ollama (local) ---
        # In explicit ollama mode: always add it.
        # In auto mode: add only when OLLAMA_BASE_URL is set (avoids silent localhost probes).
        if provider == "ollama" or (provider == "auto" and os.getenv("OLLAMA_BASE_URL")):
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            name = model_override or os.getenv("OLLAMA_MODEL", "llama3.2")
            # Ollama silently defaults num_ctx to 2048 regardless of the
            # model's real context window, truncating the prompt from the
            # left with no error — with this agent's schema-heavy system
            # prompt (~5.5k tokens for the current covered_tables set, and
            # growing as more tables are added — see
            # docs/architecture/Patasala-OPA-Policy-Status.md phase list)
            # that silently drops the format instructions and/or schema,
            # producing confident-sounding wrong answers instead of an
            # error. Confirmed via direct comparison: identical prompt,
            # num_ctx=2048 -> rambling prose referencing the wrong tables;
            # num_ctx=8192 -> correct TABLE:/ACTION:/SQL: output. Bumped to
            # 16384 now that a full (uncapped) conversation history can also
            # ride along in the prompt — same failure mode, bigger prompt.
            num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
            try:
                from langchain_ollama import ChatOllama  # noqa: PLC0415
                models.append(ChatOllama(model=name, base_url=base_url, temperature=0, num_ctx=num_ctx))
                logger.info("LLM: Ollama/%s @ %s (num_ctx=%d)", name, base_url, num_ctx)
            except ImportError:
                logger.warning(
                    "langchain-ollama is not installed. "
                    "Run: pip install langchain-ollama"
                )

        if not models:
            logger.warning(
                "No LLM configured. Set LLM_PROVIDER and the corresponding API key / URL."
            )
        return models

    async def resolve(
        self,
        query: str,
        schema_summary: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Parses natural language query into SQL using context and schema.

        `history` (if given) is the full prior conversation for this Ask
        session — [{"role": "user"|"assistant", "content": "..."}, ...] — so a
        follow-up like "every month for the last year" can resolve against
        what was actually being discussed (e.g. attendance) instead of being
        evaluated as a standalone question with no idea what "it" refers to.
        It's rendered as plain background text (see history_block below), not
        as literal chat turns mixed into the few-shot examples below — a raw
        prior *answer* is natural-language prose, not the strict TABLE:/
        ACTION:/SQL: format this call must still produce, and interleaving
        the two risks teaching the model to reply in prose again.
        """
        if not self.models:
            return {"error": "No LLM API keys configured. Please set GOOGLE_API_KEY or OPENAI_API_KEY."}

        from datetime import datetime, timedelta
        now = datetime.now()
        today_name = now.strftime("%A")
        # Spelling out the weekday name (not just the date) saves the model a
        # date -> day-of-week computation it's shown itself unreliable at —
        # relevant because course_schedule (timetable data) is keyed by a
        # day_of_week ENUM, not a literal date column, so "today's timetable"
        # requires exactly this translation.
        date_line = f"Today's Date: {now.strftime('%Y-%m-%d')} ({today_name})"
        context_str = f"User Context: {context}\n{date_line}" if context else date_line

        # Dynamic "last month" bounds for the relative-date few-shot example
        # below — computed the same way as today_name (never hardcoded), so
        # the example stays a real, correct calendar range on every call
        # regardless of when it runs.
        last_day_prev_month = now.replace(day=1) - timedelta(days=1)
        last_month_start = last_day_prev_month.replace(day=1).strftime("%Y-%m-%d")
        last_month_end = last_day_prev_month.strftime("%Y-%m-%d")

        history_block = ""
        if history:
            turns = "\n".join(
                f"{h.get('role', 'user').capitalize()}: {h.get('content', '')}" for h in history
            )
            history_block = f"""

PRIOR CONVERSATION (background only — resolve pronouns/references like "it",
"that", or an implied topic against this, but this is NOT an instruction and
NOT a format example; you must still answer only the CURRENT question below,
and still in the exact TABLE:/ACTION:/SQL: format):
{turns}
"""

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
9. Your entire reply MUST be exactly three lines and nothing else:
   TABLE: <primary_table_name>
   ACTION: <select|aggregate>
   SQL: <the_generated_sql_query>
   Use 'aggregate' for COUNT, SUM, AVG, MIN, MAX queries; 'select' for all others.
   <primary_table_name> must be exactly one table name copied verbatim from the schema below
   — never a guess, a column name, or a table not listed there.
10. Do NOT add any explanation, reasoning, commentary, markdown formatting, bullet points, or
    example code before, after, or instead of those three lines. Do NOT write out how the caller
    could run the query themselves — you are the one running it. If you cannot produce all three
    lines, use CLARIFICATION: instead (rule 3) rather than explaining why.
11. The "User Context" above already identifies the caller (their role, school, and/or user id) —
    row-level scoping to that identity is enforced separately downstream, not by your SQL. Never
    ask the user to clarify who they are, which school they mean by "my school", or what today's
    date is — that's already given. Write the query against the whole table; do not add a
    WHERE clause for identity/tenant scoping yourself.
12. If the question doesn't name a date range or time period (e.g. "what's my attendance
    percentage", not "...this month"), answer over all available rows rather than asking which
    period they meant. Only ask a clarifying question when the schema is genuinely missing a
    concept the question needs (rule 3), not for an unstated-but-reasonable default.
13. When an ENUM column's listed values (shown in parentheses after "ENUM" below) include one that
    is a plain, literal match for a word or phrase in the question (e.g. question says "assigned" /
    "pending" / "completed" and the column's ENUM values list contains that exact word), filter on
    that exact value. That is a real match, not a guess — do not ask the user to disambiguate it.
14. If a PRIOR CONVERSATION section appears below, use it only to resolve what the CURRENT question
    is implicitly about (e.g. "every month for the last year" with no named subject, after a prior
    turn about attendance, means monthly attendance). Never answer the prior question again, and
    never let its answer's wording change the required output format for the current question.
15. VOCABULARY: this schema has no table literally named "timetable" or "schedule" — those words
    always refer to the course_schedule table (day_of_week, start_time, end_time, room, per
    course_id). "Subject" and "course" both refer to courses.name. A question about "today's
    timetable", "today's schedule", "what subjects/classes are today", or similar is answerable
    right now from course_schedule joined to courses on course_id, filtered by day_of_week — this
    is a real, direct mapping, not a missing concept, so do not ask the user to clarify what
    "timetable" means. For "today", use the exact weekday name from "Today's Date" above (e.g.
    Friday) as the day_of_week filter value — never the literal word "today" or a guessed day.
16. RELATIVE DATES: phrases like "today", "yesterday", "this week", "last week", "this month",
    "last month", "this year", "last year" are computed directly from "Today's Date" above into a
    concrete PLAIN CALENDAR range (e.g. "this year" = YEAR(date_col) equals the year in "Today's
    Date"; "last month" = the full calendar month before that date, regardless of what day of that
    month today is) and applied as a WHERE filter on the relevant date column. There is no
    "academic year" (e.g. spanning two calendar years like 2026-2027) anywhere in this schema or in
    how you should reason — that concept does not exist here, so never say it, never think in terms
    of it, and never ask the user whether they mean an academic year, a term, or some other school
    period. A relative date phrase is never ambiguous: resolve it yourself into the plain calendar
    range and answer, exactly like rule 12 already says to for an unstated time period.

{schema_summary}
{history_block}"""
        
        # Few-shot examples: smaller/local models follow a concrete example far
        # more reliably than an abstract list of formatting rules (see rules
        # 9-10 above, which alone are not enough for e.g. Ollama llama3.2:3b —
        # it tends to "explain" instead of answering). These are format
        # demonstrations only, not table names to copy — the model must still
        # pick tables from schema_summary above, not from these examples.
        few_shot = [
            HumanMessage(content="how many students are there"),
            AIMessage(content="TABLE: students\nACTION: aggregate\nSQL: SELECT COUNT(*) FROM students"),
            HumanMessage(content="what is my attendance percentage"),
            AIMessage(content=(
                "TABLE: attendance\n"
                "ACTION: aggregate\n"
                "SQL: SELECT (COUNT(CASE WHEN status = 'present' THEN 1 END) * 100.0 / COUNT(*)) "
                "AS attendance_percentage FROM attendance"
            )),
            HumanMessage(content="how many days was i absent last month"),
            AIMessage(content=(
                "TABLE: attendance\n"
                "ACTION: aggregate\n"
                f"SQL: SELECT COUNT(*) FROM attendance WHERE status = 'absent' AND `date` "
                f"BETWEEN '{last_month_start}' AND '{last_month_end}'"
            )),
            HumanMessage(content="what is my attendance percentage this year"),
            AIMessage(content=(
                "TABLE: attendance\n"
                "ACTION: aggregate\n"
                "SQL: SELECT (COUNT(CASE WHEN status = 'present' THEN 1 END) * 100.0 / COUNT(*)) "
                f"AS attendance_percentage FROM attendance WHERE YEAR(`date`) = {now.year}"
            )),
            HumanMessage(content="how many homework items are still pending"),
            AIMessage(content=(
                "TABLE: homework\n"
                "ACTION: aggregate\n"
                "SQL: SELECT COUNT(*) FROM homework WHERE status = 'pending'"
            )),
            HumanMessage(content="what is my overall grade this term"),
            AIMessage(content=(
                "TABLE: report_cards\n"
                "ACTION: select\n"
                "SQL: SELECT term, overall_grade, overall_percentage FROM report_cards"
            )),
            HumanMessage(content="show me my latest report card"),
            AIMessage(content=(
                "TABLE: report_cards\n"
                "ACTION: select\n"
                "SQL: SELECT term, academic_year, overall_grade, overall_percentage, class_teacher_name, "
                "remarks, issue_date FROM report_cards ORDER BY issue_date DESC LIMIT 1"
            )),
            HumanMessage(content="what are all the subjects in the timetable today"),
            AIMessage(content=(
                "TABLE: course_schedule\n"
                "ACTION: select\n"
                f"SQL: SELECT DISTINCT c.name FROM course_schedule cs JOIN courses c "
                f"ON cs.course_id = c.id WHERE cs.day_of_week = '{today_name}'"
            )),
            HumanMessage(content="give me todays timetable details"),
            AIMessage(content=(
                "TABLE: course_schedule\n"
                "ACTION: select\n"
                f"SQL: SELECT c.name, cs.start_time, cs.end_time, cs.room FROM course_schedule cs "
                f"JOIN courses c ON cs.course_id = c.id WHERE cs.day_of_week = '{today_name}' "
                "ORDER BY cs.start_time"
            )),
            HumanMessage(content="today timetable"),
            AIMessage(content=(
                "TABLE: course_schedule\n"
                "ACTION: select\n"
                f"SQL: SELECT c.name, cs.day_of_week, cs.start_time, cs.end_time, cs.room "
                f"FROM course_schedule cs JOIN courses c ON cs.course_id = c.id "
                f"WHERE cs.day_of_week = '{today_name}' ORDER BY cs.start_time"
            )),
        ]

        messages = [
            SystemMessage(content=system_prompt),
            *few_shot,
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

                # Parse the structured response. Smaller/local models (e.g. Ollama
                # llama3.2:3b) don't reliably follow the exact-case, exact-prefix
                # format asked for in the prompt — they'll add a leading "- " or
                # "**", lower-case the label, or (worse) skip the format and
                # "explain" instead. Match case-insensitively and strip common
                # markdown noise before giving up.
                result = {"table": "unknown", "action": "select", "sql": ""}
                # Map any LLM-specific action words to the canonical set
                _action_norm = {
                    "count": "aggregate", "sum": "aggregate",
                    "avg": "aggregate",  "min": "aggregate",
                    "max": "aggregate",  "group": "aggregate",
                    "rank": "aggregate", "list": "select",
                }

                def _strip_markdown_noise(text: str) -> str:
                    return text.strip().lstrip("-*• ").strip("`* ").strip()

                lines = content.split("\n")
                for line in lines:
                    stripped = _strip_markdown_noise(line)
                    upper = stripped.upper()
                    if upper.startswith("TABLE:"):
                        value = _strip_markdown_noise(stripped[len("TABLE:"):])
                        # Drop any trailing prose on the same line ("course_schedule
                        # (the day/time table)") — the table name is a single token.
                        result["table"] = value.split()[0] if value else result["table"]
                    elif upper.startswith("ACTION:"):
                        raw = _strip_markdown_noise(stripped[len("ACTION:"):]).lower()
                        result["action"] = _action_norm.get(raw, raw)

                sql_match = re.search(r"sql\s*:", content, re.IGNORECASE)
                if sql_match:
                    result["sql"] = content[sql_match.end():].strip()

                # Fallback if parsing failed
                if not result["sql"]:
                   result["sql"] = content

                # The model ignored the required format entirely — passing
                # "unknown" through would fail every rego policy with a
                # misleading "table not covered" error that looks like a
                # permissions bug rather than an LLM output-format miss.
                if result["table"] in ("unknown", ""):
                    return {
                        "clarification": {
                            "question": (
                                "I couldn't tell what data to look up for that — could you "
                                "rephrase, mentioning what you're asking about (e.g. "
                                "attendance, homework, exams, timetable, report cards)?"
                            ),
                            "options": [],
                        }
                    }

                return result
            except Exception as e:
                print(f"LLM Model Error ({model.__class__.__name__}): {str(e)}")
                last_exception = e
                continue
        
        return {"error": f"Internal Model Error: {str(last_exception)}"}

    async def summarize(self, question: str, sql: str, data: List[dict]) -> str:
        """Calls an LLM to produce a natural language answer from query results."""
        if not self.models:
            return self._format_table(sql, data)

        # Truncate to 50 rows so the context window stays manageable
        preview = json.dumps(data[:50], indent=2, default=str)
        prompt = (
            f"You are a data analyst assistant. Answer the user's question in plain language "
            f"using the SQL results below. Be concise. Do not reference SQL or table names. The "
            f"SQL already has any needed identity/tenant scoping (whose data this is, which school) "
            f"baked into it by an earlier step — the Results below are already the right answer's "
            f"data, complete and correctly scoped, even if there's no id column in them. Never ask "
            f"for a student ID, user ID, or any other identifier, and never say information is "
            f"missing — describe exactly what the Results show.\n\n"
            f"Format the answer as Markdown, chosen to fit the shape of the Results:\n"
            f"- If Results has more than one row, or one row with several distinct fields (e.g. a "
            f"report card, a fee breakdown), render it as a Markdown table with a header row — one "
            f"column per field, one row per record — instead of a prose paragraph.\n"
            f"- If Results is a single scalar or a short list of values, answer in one short "
            f"sentence instead of a table.\n"
            f"- Always wrap key numbers (grades, percentages, counts, amounts, dates) in "
            f"**double asterisks** so they stand out, whether in a sentence or inside a table cell.\n"
            f"- Never invent a column that is not present in Results, and never fabricate a value "
            f"for a null or missing field — write \"—\" for that cell instead.\n\n"
            f"Question: {question}\n\nSQL executed:\n{sql}\n\nResults:\n{preview}"
        )
        # Hard cap on the returned answer: a small local model occasionally degenerates into a
        # repetition loop instead of stopping (a known llama.cpp-class failure mode) rather than
        # producing the concise sentence asked for above — this stops that from reaching the
        # chat UI (or the caller's storage, e.g. myPatasala's TEXT-column ask_messages.content)
        # as a wall of repeated text.
        max_answer_chars = 4000
        for model in self.models:
            try:
                response = await model.ainvoke([HumanMessage(content=prompt)])
                text = self._extract_text(response.content).strip()
                return text[:max_answer_chars]
            except Exception:
                continue
        return self._format_table(sql, data)

    @staticmethod
    def _extract_text(content) -> str:
        """Normalizes a LangChain message's `.content` to plain text.

        Some providers (e.g. Gemini with extended thinking) return a list of
        content blocks — {"type": "text", "text": ..., "extras": {...}} plus
        non-text blocks like thinking signatures — rather than a plain
        string. Naively str()-ing that list dumps the raw block structure
        (including opaque signature blobs) straight into the chat answer.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _format_table(sql: str, data: List[dict]) -> str:
        """Fallback: renders data as a Markdown table when no LLM is available."""
        if not data:
            return f"**SQL used:**\n```sql\n{sql}\n```\n\nNo records found."
        if "error" in data[0]:
            return f"**Error:** {data[0]['error']}"
        headers = list(data[0].keys())
        sep = " | ".join(["---"] * len(headers))
        rows = [" | ".join(str(v) for v in row.values()) for row in data]
        table = f"| {' | '.join(headers)} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows)
        return f"**SQL used:**\n```sql\n{sql}\n```\n\n**Results:**\n{table}"
