from pydantic import BaseModel, Field
from typing import List, Optional

class Message(BaseModel):
    role: str = Field(..., description="Must be 'user' or 'assistant'")
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str = Field(..., description="Exact assessment name from catalog")
    url: str = Field(..., description="Verified direct canonical catalog URL")
    test_type: str = Field(..., description="Single-character categorizer: e.g., 'K' or 'P'")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Conversational text output directly addressing the user")
    recommendations: List[Recommendation] = Field(default=[], description="Array of 1 to 10 valid recommendations. MUST be empty if clarifying or refusing.")
    end_of_conversation: bool = Field(default=False, description="Set to true ONLY when the conversation objective is explicitly fulfilled.")

# Internal schema for the Intent Router stage
class RouterOutput(BaseModel):
    intent: str = Field(..., description="Must strictly be one of: CLARIFY, RECOMMEND, REFINE, COMPARE, REFUSE")
    search_query: str = Field(default="", description="Optimized search phrase extracted for vector retrieval, empty if CLARIFY/REFUSE")
