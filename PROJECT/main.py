from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from ai.assistant import handle_request

app = FastAPI()

# VERY IMPORTANT (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request format
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"   # important for multi-user

# Response format
class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = handle_request(req.message, req.session_id)
    return {"reply": response}


@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}