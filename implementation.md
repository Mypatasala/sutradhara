# Implementation Plan - Core Query Lifecycle & API

## Goal Description
Implement the core `Ask` endpoint and the "Query Lifecycle" orchestration. This means receiving a natural language question, identifying intent, securing it with OPA policies, generating legitimate SQL via proper semantic mapping, executing it via MindsDB, and summarizing the result.

## Phase 2: Ambiguity Resolution (Interactive Clarification)
The user requires an interactive agent that can detect ambiguity (e.g., unclear table/column references) and ask clarifying questions with options.

**Goal:** If the `Intent` or `Schema` resolution step has low confidence, prompt the user for options instead of guessing.

### Workflow Changes
1.  **Ambiguity Detection:**
    - In `Intent Resolution` or `Schema Map` nodes, if multiple candidates strictly match or confidence < threshold, return a `Clarification` object.
    - **Example:** User asks "Show sales". Agent detects tables `offline_sales` and `online_sales`.
2.  **API Response Update:**
    - Return type can now be `answer` OR `clarification`.
    - `clarification` payload: `{"question": "Which sales?", "options": ["Offline Sales", "Online Sales"]}`.
3.  **User Selection:**
    - User replies with a selected option.
    - API re-injects this context back into the agent to resolve the ambiguity and proceed.

---

## User Requirements Breakdown & Tech Stack
The following map shows the technology selected for each step:

| pipeline Stage | Requirement | Tool / Technology |
| :--- | :--- | :--- |
| **1. Input** | `/api/v1/ask` (POST) endpoint with JWT. | **FastAPI**, **Pydantic**, **Python-Jose**. |
| **2. Intent** | Identify intent from NL question. | **Open Data QnA** / **OpenAI GPT-4**. |
| **3. Policy** | Enforce RBAC/ABAC on the intent. | **Open Policy Agent (OPA)** via `opa-python-client`. |
| **4. Schema** | Identify relevant columns & tables. | **LangChain** (SQLDatabaseChain) + **Pydantic**. |
| **5. Ambiguity** | **(Phase 2) Detect ambiguity & generate options.** | **LangGraph** (Conditional Edges) + **Pydantic** (Clarification Schema). |
| **6. Query Ops** | Identify Where/Join/Group/Limit clauses. | **LangGraph** (State management). |
| **7. Generation** | Generate dialect-specific SQL. | **LangChain** (SQLAlchemy wrapper). |
| **8. Execution** | Execute SQL securely. | **MindsDB SDK** or **SQLAlchemy**. |
| **9. Summary** | Summarize results for the user. | **OpenAI GPT-4** via **LangChain**. |

## Proposed Architecture

### 1. API Layer
**File:** `src/gateway/routes/ask.py`
- **Tech:** `FastAPI`, `APIRouter`, `Depends` (for Auth).
- **Endpoint:** `POST /api/v1/ask`
- **Responsibility:**
    - Extract Context.
    - **Handle New Response Type:** Support returning `ClarificationResponse` alongside `AnswerResponse`.

### 2. Orchestration Layer (The "Brain")
**File:** `src/agents/query_lifecycle.py`
- **Tech:** `LangGraph`.
- **Class:** `QueryLifecycleAgent`
- **Workflow Updates:**
    1.  **Intent Resolution Phase:** ... (Same)
    
    2.  **Schema/Ambiguity Check (Phase 2):**
        - **Check:** `if len(candidate_tables) > 1 and not explicit: return Clarification`.
        - **Conditional Edge:** In LangGraph, add a conditional edge after `identify_tables`.
            - If `ambiguous` -> End and return `Clarification`.
            - If `clear` -> Proceed to `Semantic Query Construction`.
    
    3.  **Semantic Query Construction:** ... (Same)
    4.  **SQL Generation:** ... (Same)
    5.  **Execution:** ... (Same)
    6.  **Summarization:** ... (Same)

## Detailed Step-by-Step Implementation

### Step 1: Initialize Project
- **Tech:** `poetry`/`pip`, `git`.
- **Dependencies:** `fastapi`, `uvicorn`, `langchain`, `langgraph`, `openai`, `mindsdb_sdk`, `pytest`, `python-jose`, `requests`.

### Step 2: API Skeleton
- Create `src/gateway/main.py`.
- Define Pydantic models:
  ```python
  class ResponseType(str, Enum):
      ANSWER = "answer"
      CLARIFICATION = "clarification"

  class ClarificationOption(BaseModel):
      label: str
      value: str

  class AskResponse(BaseModel):
      type: ResponseType
      answer: Optional[str] = None
      clarification: Optional[ClarificationPayload] = None
  ```

### Step 3: Orchestrator Stub
- Create `src/agents/query_lifecycle.py` with `LangGraph`.
- Define nodes: `intent`, `policy`, `schema`, `clarification_check`, `sql`, `execute`.

### Step 4: Intent & Policy Integration
- Implement `identify_intent` and `enforce_policy`.

### Step 5: (Phase 2) Ambiguity Logic
- Implement logic in `identify_tables` node to detect multiple fuzzy matches.
- Add Conditional Logic in LangGraph to route to a "Return Clarification" end state if needed.

### Step 6: SQL Generation & Execution
- Implement `generate_sql` and `execute` phases.

## Verification Plan
- **Test Ambiguity:**
    - Input: "Show sales".
    - Mock Schema: `[online_sales, retail_sales]`.
    - Expected Output: `type: clarification`, `options: ["Online Sales", "Retail Sales"]`.
- **Test Resolution:**
    - Input: "Show sales" + `context: {selected_table: "online_sales"}`.
    - Expected Output: `type: answer`, `sql: SELECT * FROM online_sales`.
