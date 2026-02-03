from fastapi import FastAPI, Depends, HTTPException
from .models import AskRequest, AskResponse, ResponseType
from ..agents.query_lifecycle import QueryLifecycleAgent

app = FastAPI(
    title="Sutradhara API",
    description="Generic, Domain-Agnostic AI Data Access Platform",
    version="0.1.0"
)

# Initialize the orchestrator
orchestrator = QueryLifecycleAgent()

@app.get("/")
async def root():
    return {"message": "Welcome to Sutradhara API"}

@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # Call the LangGraph orchestrator
    result = await orchestrator.run(request.query, request.context)
    
    # Map AgentState to AskResponse
    if result.get("clarification"):
        return AskResponse(
            type=ResponseType.CLARIFICATION,
            clarification=result["clarification"]
        )
    
    return AskResponse(
        type=ResponseType.ANSWER,
        answer=result.get("answer", "No answer generated.")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
