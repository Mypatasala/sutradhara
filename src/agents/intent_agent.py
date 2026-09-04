import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..retrieval.schema_provider import SchemaProvider
from .query_plan import QueryPlan, clear_incoherent_ranking_fields

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
    WHERE clause for identity/tenant scoping yourself. This applies even when the question is
    phrased as "my profile"/"my details"/"about me" against a table (like `users`) that has an
    obvious `id` column — do NOT write `WHERE id = '<value>'` using the id from User Context, and
    never invent/guess a uuid or other identifier literal that does not appear verbatim,
    character-for-character, in the schema's sample values below. A fabricated id filter silently
    returns zero rows instead of the real answer, which is worse than no filter at all.
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
            # "my profile"/"about me" against a `users`-shaped table is the one
            # case where rule 11 (never add your own identity WHERE clause) is
            # hardest for a small model to follow, because the table has an
            # obvious `id` column and User Context literally mentions a user
            # id/email — the failure mode observed is inventing a plausible-
            # looking but WRONG uuid literal for `id`, silently returning zero
            # rows instead of the real answer. The correct move, exactly like
            # every self-scoped table above, is to leave the table
            # unconstrained and let policy do the scoping downstream.
            HumanMessage(content="show me my profile details"),
            AIMessage(content=(
                "TABLE: users\n"
                "ACTION: select\n"
                "SQL: SELECT id, first_name, last_name, email, phone, department FROM users"
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
            # class_sections joins students.section_id -> class_sections.id
            # (NOT class_sections.section_id -- that column doesn't exist).
            # class_sections.name alone is NOT a unique label -- section
            # letters ("A", "B") repeat across different grades (e.g. "3rd
            # Grade" has an "A" section and so does "1st Grade"), so grouping
            # by class_sections.name/id alone and showing only the letter
            # made a real answer misleadingly collapse multiple distinct
            # classes down to what looked like just two rows. Always join
            # school_classes (the grade/level table, via class_sections
            # .school_class_id) and combine both names into one unambiguous
            # label -- every class_sections row this joins to has exactly one
            # school_classes parent, so this never fans out extra rows.
            HumanMessage(content="how many students are in each class"),
            AIMessage(content=(
                "TABLE: class_sections\n"
                "ACTION: select\n"
                "SQL: SELECT CONCAT(school_classes.name, ' - ', class_sections.name) AS class_name, "
                "COUNT(*) AS class_size FROM class_sections "
                "JOIN school_classes ON class_sections.school_class_id = school_classes.id "
                "JOIN students ON students.section_id = class_sections.id "
                "GROUP BY class_sections.id, school_classes.name, class_sections.name"
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

    _STRUCTURED_SYSTEM_PROMPT = """You are choosing a structured query plan for a school-data question -- NOT writing SQL.
{context_str}

You may only choose from the following closed vocabulary. Do not invent table names, column
names, joins, or SQL of any kind -- deterministic code owns all of that; you only choose intent.

ENTITIES -- pick the entity based on WHAT IS BEING MEASURED, not which noun the question happens
to use as its grammatical subject. "Which STUDENTS have the lowest attendance" is a question about
attendance (the thing being measured and ranked), not about the students entity itself -- use
entity=attendance, NEVER entity=students, whenever a question is ranking/measuring students by
some quantity (attendance, homework, grades, etc.) rather than asking about student identity/
roster details themselves:
- students -- pupils. Supports: count, list. Can be grouped by_class (each student's class/section).
  Can filter by grade (a dynamic lookup -- grade labels are per-school data, e.g. "5" or "10" or
  "KG", not a fixed list; validated the same way subject names are).
  Only for questions about the student roster/identity itself (e.g. "list students in class 5A",
  "how many students are in each class") -- never for ranking students by a measured quantity.
  "class"/"section" is NOT its own subject here -- it only exists as a grouping dimension OF
  students (group_by=by_class), used when the question is about STUDENTS broken down by class. A
  question asking about classes THEMSELVES (not students) uses the separate school_classes entity
  below instead -- see CLASSES COUNT vs STUDENTS COUNT.

TOTAL COUNT vs BY_CLASS GROUPING -- these are DIFFERENT questions, do not default to grouping: a
plain "how many students" question with NO per-class breakdown requested means group_by is UNSET
(the default) -- it must NEVER be set to by_class just because students CAN be grouped by class.
Only set group_by=by_class when the question itself asks for a breakdown ("each class", "per
class", "by class"). Study the contrast in these three adjacent examples exactly:
  Q: "How many students are in my school?" -> entity=students, operation=count, group_by unset
     (a plain total -- "my school" scopes WHICH students via policy/context downstream, it is
     NOT a request for a per-class breakdown; never set group_by here)
  Q: "How many students are there?" -> entity=students, operation=count, group_by unset
     (same plain-total shape, no entity/class mentioned at all -> definitely no group_by)
  Q: "How many students are in each class?" -> entity=students, operation=count,
     group_by=by_class (the word "each class" is an explicit per-class BREAKDOWN request -> THIS
     is the only shape that gets group_by=by_class; no filters, no joins, no aliases -- you only
     ever choose these three fields for this question)

GRADE FILTER vs BY_CLASS GROUPING -- these are also DIFFERENT questions, do not substitute one for
the other: "students in Grade 5" names ONE specific grade -> that is a FILTER
(filters=[{{"field": "grade", "value": "5"}}]), never group_by=by_class (which produces a
breakdown across EVERY class with no specific value requested, and is invalid combined with
operation=list).
  Q: "List all students in Grade 5." -> entity=students, operation=list,
     filters=[{{"field": "grade", "value": "5"}}] (a named grade -> a FILTER, never
     group_by=by_class; no display_fields needed -- sensible defaults are used automatically)
- attendance -- daily attendance records. Supports: count, percentage, list. Can filter by status
  (present/absent/late/excused) -- but ONLY when the question actually names a status; leaving
  filters empty means EVERY status is included, never just "present". Has a date column (use
  date_range). Can group by_student (one row per student, e.g. "each student's attendance").
  operation=list shows individual attendance records (student, date, status) -- use it for
  "show"/"show me attendance" questions, never entity=students. EXCEPTION: a question that is
  RANKING/COMPARING students against each other ("lowest"/"highest"/"top N"/"bottom N", see
  RANKING below) is NEVER operation=list, even if it also says "show" -- ranking is always
  operation=percentage with group_by=by_student; extreme/sort/limit only ever combine with
  percentage (or count), never with list.
- homework -- homework assignments. Supports: count, list. Can filter by status
  (pending/submitted/graded/late).
- report_cards -- a student's own report cards. Supports: list only. Can sort by issue_date and
  limit results (e.g. "latest" = sort issue_date desc, limit 1).
- course_schedule -- the timetable (day/time/room per course). Supports: list only. Can filter by
  day_of_week, or by subject (a dynamic lookup filter -- subject names are real course names, not
  a fixed list). Can group by_subject. "timetable"/"schedule" always means this entity.
- users -- staff/self profile fields (name, email, phone, department). Supports: list only.
- school_classes -- the school's own grade-level classes (e.g. "5th Grade", "6th Grade") as
  things in their own right -- NOT students, NOT a per-student breakdown. Supports: count. Use
  this whenever the question asks how many classes/grades exist, not how many students are in
  them. A "class" in this product always means a grade level alone -- "section" (e.g. "A", "B")
  is a separate, different concept never meant by a bare "class".

OPERATIONS: count, list, percentage (requires percentage_of: the ENUM filter defining the
numerator, e.g. status=present -- the denominator is automatically every row in scope, do not
specify it separately).

GROUPING (group_by): by_class (students only), by_status, by_day_of_week, by_subject, by_term,
by_student (attendance only). Only set group_by when the question asks for a breakdown ("each
class", "per class", "by status", "each student") -- a plain "how many X" with no breakdown should
leave group_by unset.

RANKING -- "lowest/highest" vs "top/bottom N" are DIFFERENT questions, never guess a number:
- "lowest X" / "highest X" / "who has the least/most" with NO number stated in the question ->
  set extreme="lowest" or "highest". Do NOT also set sort or limit -- extreme means every row tied
  at the actual minimum/maximum, not one arbitrary row, and is invalid combined with sort/limit.
- "N lowest/highest", "top N", "bottom N" with a number the question actually states -> set
  sort={{"field": "aggregate_value", "direction": "asc" or "desc"}} and limit=N using that exact
  number, and leave extreme unset. Never invent a limit when no number was stated -- use extreme
  instead. The verb in the question ("show", "list", "who has") is irrelevant to this choice --
  only whether a number is actually stated decides extreme vs sort+limit.

WORKED EXAMPLES for ranking questions (study these exactly):
Q: "Which students have the lowest attendance?"
-> entity=attendance, operation=percentage, group_by=by_student,
   percentage_of={{"numerator": {{"field": "status", "value": "present"}}}}, extreme=lowest
   (no sort, no limit -- "lowest" with no number means every tied row)

Q: "Who has the highest attendance?"
-> entity=attendance, operation=percentage, group_by=by_student,
   percentage_of={{"numerator": {{"field": "status", "value": "present"}}}}, extreme=highest
   (group_by=by_student is REQUIRED here even though the question says "who", not "students")

Q: "Show the 5 students with the lowest attendance."
-> entity=attendance, operation=percentage, group_by=by_student,
   percentage_of={{"numerator": {{"field": "status", "value": "present"}}}},
   sort={{"field": "aggregate_value", "direction": "asc"}}, limit=5
   (a stated number -> sort+limit, NEVER extreme; extreme and sort/limit are mutually exclusive)

Q: "List the 3 students with the highest attendance."
-> entity=attendance, operation=percentage, group_by=by_student,
   percentage_of={{"numerator": {{"field": "status", "value": "present"}}}},
   sort={{"field": "aggregate_value", "direction": "desc"}}, limit=3
   (still sort+limit with the stated number, no matter what verb the question uses)

Q: "What is my attendance percentage?"
-> entity=attendance, operation=percentage, group_by unset,
   percentage_of={{"numerator": {{"field": "status", "value": "present"}}}}
   (NO extreme, NO sort, NO limit, NO group_by here -- this question names no ranking language at
   all ("lowest"/"highest"/"who has the least/most") and no stated number, it just asks for ONE
   caller's own percentage. extreme is ONLY for a question that is actually comparing/ranking
   MULTIPLE people against each other -- the word "percentage" or the entity "attendance" alone is
   never enough to set extreme; the question must be ranking-shaped, not just percentage-shaped.)

DATE_RANGE: all_time (default), today, this_week, last_week, this_month, last_month, this_year,
last_year, last_30_days. Never compute a date yourself -- always pick one of these enum values;
the actual date math happens in deterministic code.
  "the last 30 days" / "past 30 days" -> date_range=last_30_days specifically -- this is a true
  rolling 30-day window (today and the preceding 29 days), NOT the same thing as this_month or
  last_month (a calendar month can be anywhere from 28 to 31 days, and "this month" may not have
  30 days of history yet). Never substitute last_week or this_month/last_month for "last 30 days"
  -- last_30_days exists specifically so you never have to approximate it with a different window.

DISPLAY_FIELDS (for operation=list): pick only fields relevant to the entity as described above --
if unspecified, sensible defaults are used automatically.

CLASSES COUNT vs STUDENTS COUNT -- these ask about completely different things; a bare "class"
always means a grade level (school_classes), never a count of students:
  Q: "How many classes are there?" -> entity=school_classes, operation=count, group_by unset
     (asks how many CLASSES exist -- nothing to do with how many students are in them)
  Q: "How many classes are in my school?" -> entity=school_classes, operation=count, group_by
     unset (same shape; "in my school" scopes WHICH classes via policy/context downstream, it
     does NOT mean entity=students just because the word "school" appears)
  Q: "How many students are in each class?" -> entity=students, operation=count,
     group_by=by_class (asks about STUDENTS broken down per class -- entity=students, never
     entity=school_classes, since students are what's being counted)
  Q: "How many students are in my school?" -> entity=students, operation=count, group_by unset
     (a plain STUDENT total -- entity=students, never entity=school_classes, even though the
     word "school" appears in both this question and the entity name "school_classes")

ATTENDANCE: LIST vs PERCENTAGE vs COUNT -- these are three DIFFERENT questions about the same
entity; the verb decides which one, and a status filter is NEVER added unless the question
actually names a status:
  Q: "Show me attendance for the last 30 days." -> entity=attendance, operation=list,
     group_by unset, date_range=last_30_days, filters=[] (a "show"/"show me" question about
     attendance itself wants the individual RECORDS -- operation=list, not percentage or count;
     no status named -> filters stays empty, which means every status, not just "present")
  Q: "What is the attendance percentage for the last 30 days?" -> entity=attendance,
     operation=percentage, group_by unset, date_range=last_30_days,
     percentage_of={{"numerator": {{"field": "status", "value": "present"}}}} (the word
     "percentage" is what selects operation=percentage here -- percentage_of.numerator is a
     structural requirement of the percentage operation itself, not a filter the question stated)
  Q: "How many attendance records are there in the last 30 days?" -> entity=attendance,
     operation=count, group_by unset, date_range=last_30_days, filters=[] (a plain COUNT of
     every record in range -- the question never named a status, so filters stays completely
     empty; do NOT add filters=[{{"field": "status", "value": "present"}}] here just because
     "attendance" and "present" are commonly associated -- an empty filters list on a COUNT
     means ALL FOUR statuses are counted, not just present)

IF THE QUESTION IS OUT OF SCOPE (needs an entity/concept not listed above, e.g. fees, salaries,
notifications): set can_answer=false, unresolved_reason="out_of_scope", and write a
clarification_question.

IF THE QUESTION IS ABOUT SOMETHING IN SCOPE BUT YOU CANNOT CONFIDENTLY MAP IT to a specific plan
(missing a needed detail, genuinely ambiguous phrasing): set can_answer=false,
unresolved_reason="ambiguous", and write a clarification_question. Do NOT guess a plan you are not
confident in."""

    async def resolve_structured(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryPlan:
        """Structured-output resolution: constrains the model to the
        QueryPlan JSON schema via each provider's native schema-constrained
        decoding (Ollama's `format`, OpenAI/Gemini's tool-calling-based
        structured output under langchain's common `with_structured_output`
        interface) -- the model can only ever choose from closed semantic
        vocabulary, never free SQL text, table names, joins, or aliases.

        Raises RuntimeError if every configured model's structured-output
        call fails (a TECHNICAL failure, not a semantic one). Callers
        (query_lifecycle.py) must retry once then fall back to the legacy
        free-text resolve() path on this exception -- that is the only
        fallback path from a technical failure. A successfully returned
        QueryPlan with can_answer=False is NOT this exception -- see
        query_lifecycle.py's fallback decision tree for how the two are
        handled differently.

        CONTRACT (2026-09-03 Principal Engineer review): the returned
        QueryPlan is guaranteed to have coherent ranking fields --
        `extreme`/aggregate-value `sort` are never present without the
        structural precondition (group_by set + an aggregate operation)
        that makes them meaningful; see query_plan.py's
        `clear_incoherent_ranking_fields` for the full root-cause
        investigation and safety proof. This is applied HERE, at the single
        canonical parsing boundary every caller goes through --
        resolve_structured_with_feedback's retry delegates to this same
        method -- so there is exactly one QueryPlan contract regardless of
        whether a caller uses resolve_structured() directly, its retry
        path, or the full QueryLifecycleAgent pipeline. This guarantee does
        NOT extend to full semantic validity (entity/operation compatibility,
        filter values, etc.) -- that remains QueryPlanValidator's job, a
        deliberately separate concern (see this module's own docstring).
        """
        if not self.models:
            raise RuntimeError("No LLM configured for structured resolution.")

        context_str = f"User Context: {context}" if context else ""
        prompt = self._STRUCTURED_SYSTEM_PROMPT.format(context_str=context_str) + f"\n\nQuestion: {query}"

        last_exception: Optional[Exception] = None
        for model in self.models:
            try:
                structured_model = model.with_structured_output(QueryPlan)
                plan = await structured_model.ainvoke(prompt)
                if not isinstance(plan, QueryPlan):
                    raise TypeError(f"Expected QueryPlan, got {type(plan)!r}")
                return clear_incoherent_ranking_fields(plan)
            except Exception as e:
                logger.warning("Structured resolution failed on %s: %s", model.__class__.__name__, e)
                last_exception = e
                continue

        raise RuntimeError(f"Structured resolution failed on every configured model: {last_exception}")

    async def resolve_structured_with_feedback(
        self, query: str, context: Optional[Dict[str, Any]], validation_feedback: str
    ) -> QueryPlan:
        """Retry path for a plan that parsed successfully but failed
        semantic validation -- appends the validator's specific failure
        reasons as corrective feedback and asks for a corrected plan. Used
        exactly once per question (see query_lifecycle.py); a second
        failure fails closed to a clarification, never a further retry and
        never legacy fallback."""
        feedback_prompt = (
            f"Question: {query}\n\n"
            f"Your previous plan was invalid: {validation_feedback}\n"
            f"Please produce a corrected plan using only the entities/operations/groupings/filters "
            f"described above that are actually valid for this question."
        )
        return await self.resolve_structured(feedback_prompt, context)

    async def summarize(
        self, question: str, sql: str, data: List[dict], context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Calls an LLM to produce a natural language answer from query results."""
        if not self.models:
            return self._format_table(sql, data)

        # Truncate to 50 rows so the context window stays manageable
        preview = json.dumps(data[:50], indent=2, default=str)

        # Self-scoping note: policy row filters aren't always narrowed to the
        # caller alone (e.g. an ADMIN's same-school "users" access legitimately
        # returns every staff row in the school, not just their own) — unlike a
        # STUDENT's own attendance/report-card queries, where the policy filter
        # already IS self-only and Results already contains nothing else. When
        # Results can span multiple people AND the question is specifically
        # about "my own" record, the summarizer — not the SQL-generation step,
        # which is deliberately kept out of identity filtering (see
        # intent_agent's system prompt rule 11) — is the right place to pick
        # the caller's own row back out of a broader, still-correctly-
        # authorized result set. See docs/architecture
        # /Patasala-OPA-Policy-Status.md "Open item" for the incident this
        # fixes (myPatasala ADMIN's "show me my profile details").
        identity_note = ""
        if context and (context.get("user_id") or context.get("email")):
            identity_bits = ", ".join(
                f"{k}={v}" for k, v in (("id", context.get("user_id")), ("email", context.get("email")))
                if v
            )
            identity_note = (
                f"\nCaller Identity: {identity_bits}\n"
                f"If the question asks specifically about the caller's OWN record (e.g. \"my "
                f"profile\", \"my details\", \"about me\") and Results contains more than one row, "
                f"find the row whose id/email/user_id field matches Caller Identity above and answer "
                f"using ONLY that row — ignore the other rows entirely. Only say no matching record "
                f"exists if none of the rows in Results actually match. If Results already contains "
                f"a single row, or several rows that are all clearly about one person already (e.g. "
                f"daily attendance entries), this note does not apply — answer from all of Results "
                f"as usual.\n"
            )

        prompt = (
            f"You are a data analyst assistant. Answer the user's question in plain language "
            f"using the SQL results below. Be concise. Do not reference SQL or table names. The "
            f"SQL already has any needed tenant scoping (which school) baked into it by an earlier "
            f"step. Never ask for a student ID, user ID, or any other identifier, and never say "
            f"information is missing — describe exactly what the Results show (subject to the "
            f"Caller Identity note below, if present).\n{identity_note}\n"
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
