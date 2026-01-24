# Implementation Plan - Core Query Lifecycle & API

## Goal Description
Implement the core `Ask` endpoint and the "Query Lifecycle" orchestration. This means receiving a natural language question, identifying intent, securing it with OPA policies, generating legitimate SQL via proper semantic mapping, executing it via MindsDB, and summarizing the result.

## User Requirements Breakdown & Tech Stack
The user specified a detailed pipeline. The following map shows the technology selected for each step:

| pipeline Stage | Requirement | Tool / Technology |
| :--- | :--- | :--- |
| **1. Input** | `/api/v1/ask` (POST) endpoint with JWT. | **FastAPI** (Web Framework), **Pydantic** (Validation), **Python-Jose** (JWT Parsing). |
| **2. Intent** | Identify intent from NL question. | **Open Data QnA** / **OpenAI GPT-4** (via LangChain prompt). |
| **3. Policy** | Enforce RBAC/ABAC on the intent. | **Open Policy Agent (OPA)** (Rego policies) executed via `opa-python-client`. |
| **4. Schema** | Identify relevant columns & tables. | **LangChain** (SQLDatabaseChain with custom prompt) + **Pydantic** (Output Parser). |
| **5. Query Ops** | Identify Where/Join/Group/Limit clauses. | **LangGraph** (State management to build query step-by-step). |
| **6. Generation** | Generate dialect-specific SQL. | **LangChain** (SQLAlchemy wrapper). |
| **7. Execution** | Execute SQL securely. | **MindsDB SDK** (Data federation) or **SQLAlchemy** (Direct DB connector). |
| **8. Summary** | Summarize results for the user. | **OpenAI GPT-4** (or equivalent LLM) via **LangChain**. |

## Proposed Architecture

### 1. API Layer
**File:** `src/gateway/routes/ask.py`
- **Tech:** `FastAPI`, `APIRouter`, `Depends` (for Auth).
- **Endpoint:** `POST /api/v1/ask`
- **Responsibility:**
    - Parse Bearer Token using `python-jose`.
    - Extract Context: `user_id`, `roles`, `tenant_id`.
    - Pass to `QueryLifecycleAgent`.

### 2. Orchestration Layer (The "Brain")
**File:** `src/agents/query_lifecycle.py`
- **Tech:** `LangGraph` (for stateful workflow), `LangChain` (for LLM interaction).
- **Class:** `QueryLifecycleAgent`
- **Workflow:**
    1.  **Intent Resolution Phase (LangChain):**
        - Uses `ChatOpenAI` or `Open Data QnA` API.
        - Prompt: "Classify this question into an intent object..."
    
    2.  **Policy Enforcement Phase (OPA):**
        - Uses `requests` to call a local OPA sidecar or library.
        - Input: `input: { "user": ..., "intent": ... }`
        - Policy: `rego` files in `src/policy/`.

    3.  **Semantic Query Construction Phase (LangGraph API):**
        - **State:** `QueryState` (holds identified tables, filters).
        - **Nodes:**
            - `identify_tables`: LLM call to map business terms to tables.
            - `build_where_clause`: Inject mandatory filters (`tenant_id`).
            - `build_joins`: Use pre-defined schema graphs to determine joins.

    4.  **SQL Generation Phase (LangChain):**
        - Tool: `SQLDatabaseChain` or Custom Chain.
        - Input: The constructed "Abstract Query".
        - Output: Dialect-specific SQL string.
    
    5.  **Execution Phase (MindsDB):**
        - Tech: `mindsdb_sdk` or `sqlalchemy` engine.
        - Action: Run query and fetch results as JSON.

    6.  **Summarization Phase (LangChain):**
        - Tool: `LLMChain` with `SummaryPrompt`.
        - Input: Original Question + JSON Results.
        - Output: Final natural language answer.

## Detailed Step-by-Step Implementation

### Step 1: Initialize Project
- **Tech:** `poetry` or `pip`, `git`.
- Create directory structure (`src/`, `tests/`).
- **Dependencies:** `fastapi`, `uvicorn`, `langchain`, `langgraph`, `openai`, `mindsdb_sdk`, `pytest`, `python-jose`, `requests`.

### Step 2: API Skeleton
- **Tech:** `FastAPI`.
- Create `src/gateway/main.py` entry point.
- Define `AskRequest` and `AskResponse` Pydantic models.

### Step 3: Orchestrator Stub
- **Tech:** `LangGraph`.
- Create `src/agents/query_lifecycle.py`.
- Define the `StateGraph` with nodes for each phase (Intent, Policy, SQL, Execute).

### Step 4: Intent & Policy Integration
- **Tech:** `OpenAI API`, `OPA`.
- Implement `identify_intent` node.
- Implement `enforce_policy` node using a mock policy function for starters.

### Step 5: SQL Generation
- **Tech:** `LangChain SQLDatabase`.
- Connect to a dummy SQLite DB.
- Implement `generate_sql` node.

### Step 6: Execution & Summary
- **Tech:** `MindsDB`.
- Execution node mocks the MindsDB call initially (returning static data).
- Summarization node calls LLM.

## Verification Plan
- **Unit Tests:** `pytest` + `pytest-asyncio` for async API tests.
- **Integration:** Run `docker-compose up` (API + OPA + MindsDB) and hit the endpoint.
