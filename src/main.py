"""
Main FastAPI application for the JioPay RAG Chatbot.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from src.config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="JioPay RAG Chatbot API",
    description="Production-grade RAG chatbot for JioPay customer support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    include_citations: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]
    retrieved_chunks: List[dict]
    metadata: dict

class HealthResponse(BaseModel):
    status: str
    message: str

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="JioPay RAG Chatbot API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for RAG-based question answering.
    
    Args:
        request: Chat request with query and optional parameters
        
    Returns:
        Chat response with answer, citations, and metadata
    """
    try:
        # TODO: Implement RAG pipeline
        # This will be implemented in the RAG implementation phase
        
        # Placeholder response
        return ChatResponse(
            answer="RAG pipeline not yet implemented. This is a placeholder response.",
            citations=[],
            retrieved_chunks=[],
            metadata={
                "model": "placeholder",
                "processing_time": 0.0,
                "tokens_used": 0
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available embedding and generation models."""
    return {
        "embedding_models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "intfloat/e5-base",
            "intfloat/e5-large",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5"
        ],
        "generation_models": [
            "gemini-pro",
            "gpt-3.5-turbo",
            "gpt-4"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    settings = get_settings()
    return {
        "total_documents": 0,  # TODO: Implement actual count
        "total_chunks": 0,     # TODO: Implement actual count
        "embedding_model": settings.default_embedding_model,
        "chunk_size": settings.default_chunk_size,
        "chunk_overlap": settings.default_chunk_overlap
    }

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )
