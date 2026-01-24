# Sutradhara: Comprehensive Design and Requirements Document

## 1. Introduction
**Sutradhara** is a generic, domain-agnostic AI data access platform designed to bridge natural language user queries and structured relational databases. The platform enables users to retrieve meaningful insights from enterprise data using plain language while enforcing strict governance, access control, and security boundaries.

Sutradhara is designed for multi-domain usage including:
- Education
- Insurance
- Ecommerce
- Inventory
- HR
- Finance systems

## 2. Problem Statement
Modern organizations store large volumes of data in relational databases that are difficult to access for non-technical users. Existing solutions either expose raw data insecurely or tightly couple AI logic to database schemas.

**Sutradhara solves this by introducing a layer that is:**
- **Policy-driven:** Access is controlled by explicit rules, not just database permissions.
- **Schema-aware:** Understands the semantic meaning of data, not just table structures.
- **LLM-assisted:** Uses AI for understanding intent and summarizing results, but not for direct execution.

This ensures safe and explainable data access.

## 3. Vision and Objectives
The vision of Sutradhara is to act as an intelligent orchestrator that understands user intent, enforces organizational rules, and delivers precise, summarized answers.

### Key Objectives
- **Domain Independence:** Can be applied to any industry vertical via configuration.
- **Zero-trust AI Interaction:** Never trust the model implicitly; validate all outputs.
- **Configuration-driven Onboarding:** Easy setup for new domains without code changes.
- **Enterprise-grade Observability:** Full visibility into operations, decisions, and data flows.

## 4. Core Design Principles
1. **Zero Direct Database Access by LLMs:** The AI never executes SQL directly. It functions as a translator, not an executor.
2. **Strict Separation of Concerns:** Decoupled architecture where each component (intent, policy, retrieval) is independent.
3. **Metadata-driven Domain Configuration:** Flexible schema definitions allows the system to adapt to any database back-end.
4. **Least-privilege Data Exposure:** Only essential data is retrieved and passed to the summarization layer.
5. **Full Auditability:** Track every decision, from intent resolution to final SQL query and data delivery.

## 5. High-Level Architecture
The architecture consists of multiple decoupled layers, each performing a single responsibility and communicating via well-defined contracts:

- **API Gateway:** Entry point for all user requests.
- **Identity and Context Resolver:** Enriches requests with user identity and organizational context.
- **Intent Resolution Agent:** detailed intent understanding using LLMs.
- **Semantic Mapping Engine:** Maps logical concepts to physical database schemas.
- **Policy and Authorization Engine:** The "Brain" of security, creating boundaries for what is allowed.
- **Data Retrieval Engine:** Safe SQL builder and executor.
- **LLM Summarization Layer:** detailed response generation based on retrieved data.

## 6. Identity and Context Management
User identity is established using **JWT tokens** that contain:
- User identifiers (sub)
- Roles (scopes/groups)
- Organization context (tenant_id)
- Linked entities (e.g., student_id for a parent)

This context persists throughout request processing and is used to enforce row-level and column-level security policies.

## 7. Intent Resolution Layer
This layer uses an LLM to transform natural language questions into structured **intent objects**.
The output format is strictly validated JSON to avoid ambiguity.

**Intent definitions include:**
- **Requested Action:** (e.g., "Summarize", "List", "Compare")
- **Logical Entity:** (e.g., "Attendance", "Invoices", "Grades")
- **Filters:** (e.g., "last month", "status=pending")
- **Temporal Constraints:** (e.g., "between 2023-01-01 and 2023-12-31")

## 8. Schema and Semantic Configuration
Domains are onboarded by registering database connections, schemas, tables, and column metadata.

**Logical Entities** are mapped to physical tables through configuration files that define:
- **Semantic Meaning:** What the data represents (e.g., "Student Grade" maps to `t_results`).
- **Ownership Relationships:** How to link a user to this data (e.g., `t_results.student_id` -> `user.linked_student_id`).
- **Join Paths:** Pre-defined safe join paths to avoid cartesian products.

## 9. Policy and Access Control Engine
The policy engine evaluates whether a resolved intent is allowed for the requesting user. Unauthorized access attempts are blocked *before* data retrieval.

**Policies define:**
- **Role-based Access:** "Can 'Parents' view 'Grades'?"
- **Entity-level Permissions:** "Can this user access data for School ID 101?"
- **Column Visibility:** "Hide PII columns like SSN or Phone Number."
- **Row-level Filtering Rules:** "Append `WHERE student_id = :user_student_id` to all queries."

## 10. Data Retrieval Engine
This engine generates safe, parameterized SQL queries based on validated intent and policy outputs.

**Capabilities:**
- **Query Construction:** Builds SQL usage safe templates or ORM layers.
- **Result Limiting:** Enforces strict limits on result size to prevent data dumping.
- **Masking:** Masks sensitive columns at the database level where possible.
- **Authorization Enforcement:** Ensures only authorized data is retrieved.

## 11. LLM Summarization Layer
The summarization layer receives only **sanitized datasets** along with the original user question.
The LLM generates human-readable summaries, explanations, or insights **without** exposure to:
- Internal database identifiers (PKs/FKs)
- Physical schema names
- Unresolved raw data

## 12. Multi-Domain Usage Examples
- **Education:** Parents viewing student attendance in school systems.
- **Insurance:** Customers reviewing insurance policies and claim status.
- **Ecommerce:** Shoppers checking order history and delivery estimates.
- **Inventory:** Managers inspecting inventory levels and low-stock alerts.

## 13. Open Source Tools and Ecosystem Strategy
Sutradhara leverages existing open-source innovations as building blocks while providing the necessary enterprise orchestration layer.

### Natural Language Database Interaction
These tools handle the core "Text-to-SQL" or "Text-to-Query" logic.
*   **Open Data QnA (Google Cloud):**
    *   *Features:* Python library for chatting with SQL databases, multi-turn conversations, modular architecture.
    *   *Role:* Can be used as the core "Intent Resolution" agent.
*   **NLP-to-SQL (Community Project):**
    *   *Features:* Open-source SQL interface for PostgreSQL/MySQL, schema aware.
    *   *Role:* Useful reference for safe SQL generation patterns.

### Data Platforms & Connectors
*   **MindsDB Open Source:**
    *   *Features:* Connects to 200+ data sources, enables SQL queries over AI models.
    *   *Role:* Provides the "Data Retrieval Engine" connectivity layer, abstracting specific DB drivers.

### Text2SQL Frameworks
These serve as specialized engines for translating intent to structured queries.
*   **QueryWeaver:** Graph-powered engine with a semantic layer.
*   **PremSQL:** Modular tools for NL2SQL evaluation.
*   **AGENTIQL:** Multi-agent framework for complex query planning.

## 14. Gap Analysis: Build vs. Buy
While open-source tools provide excellent foundations, they lack critical enterprise features that Sutradhara implements.

| Feature | Open Source (Typical) | Sutradhara (Custom Layer) |
| :--- | :--- | :--- |
| **Security Model** | Database credentials only | **JWT-based Row-Level Security & Permission Scoping** |
| **Configuration** | Raw Schema inspection | **Semantic Entities & Ownership Mappings** |
| **Policy** | None or Basic RBAC | **Domain-agnostic Policy Enforcement Engine (OPA-style)** |
| **Summarization** | Direct LLM output | **Secure Context Wrapping & Sanitization** |
| **Tenancy** | Single Database Focus | **Multi-tenant Data Filtering & User Context Injection** |

**Sutradhara's Value Proposition:** It acts as the "Secure Orchestrator" that wraps these raw tools, injecting identity, policy, and semantic understanding that raw Text2SQL libraries miss.

## 15. Architecture Integration Strategy
How the tools fit into the Sutradhara generic design:

1.  **Intent Phase:** Use **Open Data QnA** or **LangGraph** agents to parse the user's natural language into a structured intermediate representation (not direct SQL).
2.  **Policy Phase:** Intercept the intent. Apply **Custom Logic** to inject `WHERE` clauses (e.g., `WHERE user_id = $JWT.sub`) and filter accessible columns.
3.  **Execution Phase:** Pass the *modified, secure* intent to a connector like **MindsDB** or a custom SQL builder to execute the actual fetch.
4.  **Response Phase:** Pass the clean JSON result + original question to the **LLM** for a final summarized answer.

## 16. Non-Functional Requirements
- **High Availability:** Redundant services and stateless design.
- **Horizontal Scalability:** Ability to handle increasing concurrent users.
- **Low Latency Response Times:** Caching frequent intent resolutions.
- **Strong Security Posture:** Defense in depth (Gateway -> Policy -> DB).
- **Observability:** Centralized logging of all "Intent -> SQL" transformations.
- **Compliance Readiness:** GDP/SOC2 compliance through strict access logs.

## 17. Recommended Implementation Workflow
1.  **Register the Domain:** Define the data source connection strings.
2.  **Configure Semantic Metadata:** Map physical tables to "human" concepts (Entities).
3.  **Define Policies:** Write rules for who can see what (Policy Engine).
4.  **Deploy Intent Resolution Agents:** Spin up the specific agents (e.g., Open Data QnA instances) for the domain.
5.  **Integrate LLM Provider:** Connect to OpenAI, Anthropic, or local LLMs.
6.  **Onboard End Users:** Provision JWTs and map them to domain data ownership.

## 18. Conclusion
Sutradhara provides a **robust, secure, and extensible foundation** for AI-driven access to enterprise data. By combining best-in-class open-source Text2SQL tools with a custom **governance and security layer**, it solves the "last mile" problem of safely deploying AI analysts in the enterprise.
