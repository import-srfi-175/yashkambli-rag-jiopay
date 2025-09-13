"""
Simplified RAG Pipeline with memory management for JioPay chatbot.
Handles memory constraints and provides fallback options.
"""
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

# Set memory management environment variables
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

try:
    import google.generativeai as genai
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

from src.config import get_settings
from src.vector_store.vector_manager import VectorStoreManager, RetrievalResult

settings = get_settings()


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    retrieved_chunks: List[RetrievalResult]
    citations: List[Dict[str, Any]]
    response_time: float
    retrieval_model: str


class SimplifiedRAGPipeline:
    """Simplified RAG pipeline with memory management."""
    
    def __init__(self):
        """Initialize the simplified RAG pipeline."""
        logger.info("🔄 Initializing Simplified RAG pipeline...")
        
        self.llm_model = None
        self.embedding_model = None
        self.vector_manager = None
        self.conversation_history = []
        
        self._initialize_llm()
        self._initialize_vector_manager()
        self._initialize_embedding_model()
        
        logger.info("✅ Simplified RAG pipeline initialized")
    
    def _initialize_llm(self):
        """Initialize Gemini Pro model."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY") or settings.google_api_key
            if api_key:
                genai.configure(api_key=api_key)
                self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini 1.5 Flash model initialized")
            else:
                logger.warning("Google API key not configured, using mock responses")
                self.llm_model = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Pro: {e}")
            self.llm_model = None
    
    def _initialize_vector_manager(self):
        """Initialize vector store manager."""
        try:
            self.vector_manager = VectorStoreManager()
            logger.info("✅ Vector store manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vector manager: {e}")
            self.vector_manager = None
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for query encoding."""
        try:
            # Use BGE-small as default for memory efficiency
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            logger.info("✅ BGE-small embedding model initialized for queries")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def retrieve_chunks(self, query: str, collection_name: str = "jiopay_bge-small", 
                       n_results: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant chunks from vector store."""
        if not self.vector_manager:
            logger.error("Vector manager not initialized")
            return []
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Get collection
            collection = self.vector_manager.get_collection(collection_name)
            if not collection:
                logger.error(f"Collection {collection_name} not found")
                return []
            
            # Encode query
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([query]).tolist()[0]
            else:
                logger.error("Embedding model not available")
                return []
            
            # Retrieve similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Convert to RetrievalResult objects
            retrieved_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (1 - distance)
                    score = 1 - distance
                    
                    chunk = RetrievalResult(
                        text=doc,
                        score=score,
                        metadata=metadata or {},
                        chunk_id=f"{collection_name}_{i}"
                    )
                    retrieved_chunks.append(chunk)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def generate_answer(self, query: str, retrieved_chunks: List[RetrievalResult], 
                      conversation_context: List[Dict[str, str]] = None) -> str:
        """Generate answer using Gemini Pro with retrieved context."""
        if not self.llm_model:
            # Generate mock response based on retrieved chunks
            return self._generate_mock_answer(query, retrieved_chunks)
        
        try:
            # Prepare context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(f"Context {i}: {chunk.text}")
            
            context = "\n\n".join(context_parts)
            
            # Prepare conversation history
            history_text = ""
            if conversation_context:
                for msg in conversation_context[-3:]:  # Last 3 messages
                    history_text += f"{msg['role']}: {msg['content']}\n"
            
            # Create prompt
            prompt = f"""You are a helpful JioPay customer support assistant. Your task is to synthesize the information from the provided context to answer the user's question.

Context Information:
---
{context}
---

Previous Conversation:
{history_text}

User Question: {query}

Instructions:
1.  Read the context and the user's question carefully.
2.  Formulate a concise and helpful answer that directly addresses the user's question.
3.  Do NOT quote the context directly. Summarize the information in your own words.
4.  If the context does not contain the answer, state that you couldn't find the information and, if possible, suggest what the user should look for.

Answer:"""
            
            # Generate response
            response = self.llm_model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fall back to mock response
            return self._generate_mock_answer(query, retrieved_chunks)
    
    def _generate_mock_answer(self, query: str, retrieved_chunks: List[RetrievalResult]) -> str:
        """Generate a mock answer based on retrieved chunks when LLM is not available."""
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about JioPay features, pricing, security, or support."
        
        # Extract key information from chunks
        context_info = []
        for chunk in retrieved_chunks[:3]:  # Use top 3 chunks
            context_info.append(chunk.text)
        
        # Simple template-based response
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['integrate', 'api', 'website', 'sdk']):
            return f"Based on the available information, JioPay provides easy-to-use APIs and SDKs for integration. You can integrate using REST API, Java SDK, or other available methods. Here's what I found: {' '.join(context_info[:1])[:200]}..."
        
        elif any(word in query_lower for word in ['price', 'cost', 'fee', 'plan']):
            return f"JioPay offers different pricing plans for businesses. Here are the details: {' '.join(context_info[:1])[:200]}..."
        
        elif any(word in query_lower for word in ['security', 'safe', 'encrypt', 'protect']):
            return f"JioPay provides comprehensive security features including end-to-end encryption, tokenization, and multi-factor authentication. Here's more information: {' '.join(context_info[:1])[:200]}..."
        
        elif any(word in query_lower for word in ['support', 'contact', 'help']):
            return f"JioPay offers 24/7 customer support via email, phone, and chat. Here are the contact details: {' '.join(context_info[:1])[:200]}..."
        
        elif any(word in query_lower for word in ['payment', 'method', 'card', 'upi']):
            return f"JioPay supports various payment methods including credit/debit cards, UPI, net banking, and digital wallets. Here's more information: {' '.join(context_info[:1])[:200]}..."
        
        else:
            return f"Based on the available information about JioPay: {' '.join(context_info[:1])[:300]}..."
    
    def create_citations(self, retrieved_chunks: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Create citations from retrieved chunks."""
        citations = []
        
        for chunk in retrieved_chunks:
            citation = {
                'text': chunk.text,
                'score': chunk.score,
                'chunking_strategy': chunk.metadata.get('chunking_strategy', 'unknown'),
                'model_name': chunk.metadata.get('model_name', 'unknown'),
                'source_url': chunk.metadata.get('source_url', ''),
                'source_title': chunk.metadata.get('source_title', '')
            }
            citations.append(citation)
        
        return citations
    
    def process_query(self, query: str, collection_name: str = "jiopay_bge-small", 
                     n_results: int = 5) -> RAGResponse:
        """Process a query through the RAG pipeline."""
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_chunks(query, collection_name, n_results)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_chunks, self.conversation_history)
        
        # Create citations
        citations = self.create_citations(retrieved_chunks)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep only last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        response_time = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            response_time=response_time,
            retrieval_model=collection_name
        )
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


# Alias for compatibility
RAGPipeline = SimplifiedRAGPipeline
RAGResponse = RAGResponse
