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
- Policy-driven
- Schema-aware
- LLM-assisted

This ensures safe and explainable data access.

## 3. Vision and Objectives
The vision of Sutradhara is to act as an intelligent orchestrator that understands user intent, enforces organizational rules, and delivers precise, summarized answers.

### Key Objectives
- **Domain Independence:** Can be applied to any industry vertical.
- **Zero-trust AI Interaction:** Never trust the model implicitly.
- **Configuration-driven Onboarding:** Easy setup for new domains.
- **Enterprise-grade Observability:** Full visibility into operations.

## 4. Core Design Principles
1. **Zero Direct Database Access by LLMs:** The AI never executes SQL directly.
2. **Strict Separation of Concerns:** Decoupled architecture.
3. **Metadata-driven Domain Configuration:** Flexible schema definitions.
4. **Least-privilege Data Exposure:** Only essential data is retrieved.
5. **Full Auditability:** Track every decision and data flow.

## 5. High-Level Architecture
The architecture consists of multiple decoupled layers, each performing a single responsibility and communicating via well-defined contracts:

- **API Gateway**
- **Identity and Context Resolver**
- **Intent Resolution Agent**
- **Semantic Mapping Engine**
- **Policy and Authorization Engine**
- **Data Retrieval Engine**
- **LLM Summarization Layer**

## 6. Identity and Context Management
User identity is established using **JWT tokens** that contain:
- User identifiers
- Roles
- Organization context
- Linked entities

This context persists throughout request processing and is used to enforce row-level and column-level security policies.

## 7. Intent Resolution Layer
This layer uses an LLM to transform natural language questions into structured **intent objects**.
The output format is strictly validated JSON to avoid ambiguity.

**Intent definitions include:**
- Requested action
- Logical entity
- Filters
- Temporal constraints

## 8. Schema and Semantic Configuration
Domains are onboarded by registering database connections, schemas, tables, and column metadata.

**Logical Entities** are mapped to physical tables through configuration files that define:
- Semantic meaning
- Ownership relationships
- Join paths

## 9. Policy and Access Control Engine
The policy engine evaluates whether a resolved intent is allowed for the requesting user. Unauthorized access attempts are blocked *before* data retrieval.

**Policies define:**
- Role-based access
- Entity-level permissions
- Column visibility
- Row-level filtering rules

## 10. Data Retrieval Engine
This engine generates safe, parameterized SQL queries based on validated intent and policy outputs.

**Capabilities:**
- Enforces limits on result size.
- Masks sensitive columns.
- Ensures only authorized data is retrieved.

## 11. LLM Summarization Layer
The summarization layer receives only **sanitized datasets** along with the original user question.
The LLM generates human-readable summaries, explanations, or insights **without** exposure to internal identifiers or schemas.

## 12. Multi-Domain Usage Examples
- **Education:** Parents viewing student attendance in school systems.
- **Insurance:** Customers reviewing insurance policies.
- **Ecommerce:** Shoppers checking order history.
- **Inventory:** Managers inspecting inventory levels.

## 13. Open Source Tools and Ecosystem
Sutradhara can be implemented using open-source components such as:
*   **Open Data QnA:** For intent-to-SQL workflows.
*   **MindsDB Open Source:** For data connectors.
*   **LangGraph:** For agent orchestration.
*   **Open Policy Agent (OPA):** For policy enforcement.
*   **Apache Superset:** Metadata concepts for schema management.

## 14. Non-Functional Requirements
- **High Availability**
- **Horizontal Scalability**
- **Low Latency Response Times**
- **Strong Security Posture**
- **Observability:** Logging and metrics.
- **Compliance Readiness**

## 15. Recommended Implementation Workflow
1.  Register the domain.
2.  Configure schema metadata.
3.  Define policies.
4.  Deploy intent resolution agents.
5.  Integrate an LLM provider.
6.  Onboard end users.

## 16. Conclusion
Sutradhara provides a **robust, secure, and extensible foundation** for AI-driven access to enterprise data. Its generic and policy-driven architecture makes it suitable for a wide range of industries and use cases.
