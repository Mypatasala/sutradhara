# Implementation Plan - Core Query Lifecycle & API

## Goal Description
Implement the core `Ask` endpoint and the "Query Lifecycle" orchestration. This means receiving a natural language question, identifying intent, securing it with OPA policies, generating legitimate SQL via proper semantic mapping, executing it via MindsDB, and summarizing the result.

## User Requirements Breakdown
The user specified a detailed pipeline:
1.  **Input:** `/api/v1/ask` (POST) with `{"question": "..."}` and JWT context.
2.  **Intent ID:** Identify intent (Open Data QnA) + Enforce Policy (OPA).
3.  **Schema ID:** Identify columns/tables.
4.  **Query Construction:** Identify Where/Join/Group/Order/Limit clauses.
5.  **Generation:** Generate SQL (LangChain).
6.  **Execution:** Execute SQL (MindsDB).
7.  **Summarization:** Return result to LLM for final answer.

## Proposed Architecture

### 1. API Layer
**File:** `src/gateway/routes/ask.py`
- **Endpoint:** `POST /api/v1/ask`
- **Responsibility:**
    - Validate JWT.
    - Extract User Context (User ID, Role, Tenant ID).
    - Pass Question + Context to the Orchestrator.
    - Return final JSON response.

### 2. Orchestration Layer (The "Brain")
**File:** `src/agents/query_lifecycle.py`
- **Class:** `QueryLifecycleAgent`
- **Workflow:**
    1.  **Intent Resolution Phase:**
        - Call LLM/OpenDataQnA to classify: "What is the user asking for?" (e.g., `SUM(revenue) WHERE year=2022`).
        - **Output:** Structured Intent Object.
    
    2.  **Policy Enforcement Phase:**
        - Input: Structured Intent + User Context (JWT).
        - Action: Check against OPA policies.
        - **Output:** Approved Intent (potentially modified with mandatory filters like `tenant_id=123`).

    3.  **Semantic Query Construction Phase:**
        - **Column/Table Identification:** Map "revenue" -> `sales_table.amount`.
        - **Clause Generation:**
            - *Where:* `year = 2022` AND `tenant_id = 123` (injected).
            - *Join:* `sales_table` JOIN `products` ON ...
            - *Group/Order/Limit:* As defined by intent.
        - **Output:** Abstract SQL Representation.

    4.  **SQL Generation Phase:**
        - Tool: LangChain SQLDatabaseChain (customized).
        - Action: Convert Abstract Rep -> Dialect-specific SQL (Postgres/Snowflake).
    
    5.  **Execution Phase:**
        - Tool: MindsDB / SQLAlchemy.
        - Action: Execute SQL safely.
        - **Output:** Raw Data (List of Dicts).

    6.  **Summarization Phase:**
        - Input: User Question + Raw Data.
        - Action: LLM generates text response.
        - **Output:** Final string answer.

## Detailed Step-by-Step Implementation

### Step 1: Initialize Project
- Create directory structure (`src/`, `tests/`).
- Define `requirements.txt` (fastapi, uvicorn, langchain, openai, mindsdb_sdk, pytest).

### Step 2: API Skeleton
- Create FastAPI app `src/gateway/main.py`.
- Define the `AskRequest` and `AskResponse` Pydantic models.

### Step 3: Orchestrator Stub
- Create `src/agents/query_lifecycle.py`.
- Implement the `run()` method with placeholders for each phase to ensure the pipeline flows correctly before adding complex logic.

### Step 4: Intent & Policy Integration
- Implement `_identify_intent` using a prompt template.
- Implement `_enforce_policy` as a mock function (later connect to OPA).

### Step 5: SQL Generation (LangChain)
- Set up a dummy SQLite database for testing.
- Connect LangChain to generate valid SQL from the intent.

### Step 6: Execution & Summary
- mock the MindsDB connection for initial testing.
- Implement the final LLM summarization call.

## Verification Plan
- **Unit Tests:** `tests/test_flow.py` (Mocking LLM calls).
- **Integration Test:** `curl localhost:8000/api/v1/ask` -> Should return a mocked logical answer.
