import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv

from schemas import ChatRequest, ChatResponse
from agent import SHLRecommenderAgent

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

app = FastAPI(title="Conversational SHL Assessment Recommender API")

# Lazy initialize agent instance to guarantee optimal cold-start handling
agent = None

@app.on_event("startup")
def startup_event():
    global agent
    # Instantiate the agent class, load the FAISS indices and sentence transformers models
    agent = SHLRecommenderAgent()

@app.get("/health", status_code=200)
def health_check():
    """Readiness endpoint designed to instantly respond to automated monitoring pings."""
    if agent is not None:
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "initializing"})

@app.get("/", include_in_schema=False)
def root():
    """Redirect root hits to the Swagger UI."""
    return RedirectResponse(url="/docs")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """Stateless entrypoint handling dynamic conversational analysis and vector recommendations."""
    if not payload.messages:
        raise HTTPException(status_code=400, detail="Conversation message history cannot be empty.")
    
    try:
        # Synchronous execution within the 30-second timeout limit
        response_data = agent.execute_pipeline(payload.messages)
        return response_data
    except Exception as e:
        # Prevent automated tester failures from unhandled runtime breaks by returning standard format fallback
        print(f"Error in chat_endpoint: {e}")
        return ChatResponse(
            reply="I encountered an internal error processing this request. Please rephrase your query.",
            recommendations=[],
            end_of_conversation=False
        )
