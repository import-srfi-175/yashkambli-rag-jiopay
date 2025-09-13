"""
RAG (Retrieval-Augmented Generation) implementation for JioPay chatbot.
Integrates vector retrieval with Gemini Pro for answer generation and citations.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.vector_store.vector_manager import VectorStoreManager, RetrievalResult
from src.utils import save_json, load_json, get_timestamp

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RAGResponse:
    """Represents a RAG response with citations."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievalResult]
    citations: List[Dict[str, Any]]
    model_used: str
    retrieval_model: str
    response_time: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """Complete RAG pipeline implementation."""
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.llm_model = None
        self.embedding_model = None
        self.conversation_history = []
        self._initialize_llm()
        self._initialize_embedding_model()
    
    def _initialize_llm(self):
        """Initialize Gemini Pro model."""
        try:
            if settings.google_api_key and settings.google_api_key != "your_google_api_key_here":
                genai.configure(api_key=settings.google_api_key)
                self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini 1.5 Flash model initialized")
            else:
                logger.warning("Google API key not configured, using mock responses")
                self.llm_model = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Pro: {e}")
            self.llm_model = None
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for query encoding."""
        try:
            # Use BGE-small as default for speed
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            logger.info("✅ BGE-small embedding model initialized for queries")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
    
    def retrieve_relevant_chunks(self, query: str, collection_name: str = "jiopay_bge-small", 
                               n_results: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        try:
            # Extract model name from collection
            model_name = collection_name.replace('jiopay_', '')
            
            retrieval_results = self.vector_manager.query_collection(
                collection_name, query, n_results=n_results, model_name=model_name
            )
            
            logger.info(f"Retrieved {len(retrieval_results)} chunks for query")
            return retrieval_results
            
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
                context_parts.append(f"[Source {i}] {chunk.text}")
            
            context = "\n\n".join(context_parts)
            
            # Prepare conversation history
            history_text = ""
            if conversation_context:
                history_parts = []
                for msg in conversation_context[-3:]:  # Last 3 messages
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_parts.append(f"{role}: {msg['content']}")
                history_text = "\n".join(history_parts) + "\n\n"
            
            # Create prompt
            prompt = f"""You are a helpful customer support assistant for JioPay, a digital payment platform. 
Use the provided context to answer user questions accurately and helpfully.

Previous conversation:
{history_text}

Context from JioPay documentation:
{context}

User Question: {query}

Instructions:
1. Answer the user's question based on the provided context
2. Be helpful, accurate, and professional
3. If the context doesn't contain enough information, say so politely
4. Focus on JioPay-specific information
5. Keep responses concise but informative
6. Use a friendly, supportive tone

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
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            citation = {
                'id': i,
                'chunk_id': chunk.chunk_id,
                'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                'score': chunk.score,
                'source_url': chunk.metadata.get('source_url', ''),
                'source_title': chunk.metadata.get('source_title', ''),
                'chunking_strategy': chunk.metadata.get('chunking_strategy', 'unknown'),
                'model_name': chunk.metadata.get('model_name', 'unknown')
            }
            citations.append(citation)
        
        return citations
    
    def process_query(self, query: str, collection_name: str = "jiopay_bge-small", 
                     n_results: int = 5, include_conversation: bool = True) -> RAGResponse:
        """Process a complete RAG query."""
        import time
        start_time = time.time()
        
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve_relevant_chunks(query, collection_name, n_results)
        
        if not retrieved_chunks:
            return RAGResponse(
                query=query,
                answer="I couldn't find relevant information to answer your question. Please try rephrasing or ask about JioPay features, pricing, security, or support.",
                retrieved_chunks=[],
                citations=[],
                model_used="gemini-pro",
                retrieval_model=collection_name.replace('jiopay_', ''),
                response_time=time.time() - start_time,
                metadata={'error': 'No relevant chunks found'}
            )
        
        # Generate answer
        conversation_context = self.conversation_history if include_conversation else None
        answer = self.generate_answer(query, retrieved_chunks, conversation_context)
        
        # Create citations
        citations = self.create_citations(retrieved_chunks)
        
        # Update conversation history
        if include_conversation:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
        
        response_time = time.time() - start_time
        
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            model_used="gemini-pro",
            retrieval_model=collection_name.replace('jiopay_', ''),
            response_time=response_time,
            metadata={
                'chunks_retrieved': len(retrieved_chunks),
                'conversation_length': len(self.conversation_history)
            }
        )
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()


class RAGEvaluator:
    """Evaluates RAG system performance."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def evaluate_query(self, query: str, expected_topics: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single query."""
        response = self.rag_pipeline.process_query(query)
        
        evaluation = {
            'query': query,
            'response_time': response.response_time,
            'chunks_retrieved': len(response.retrieved_chunks),
            'answer_length': len(response.answer),
            'citations_count': len(response.citations),
            'retrieval_scores': [chunk.score for chunk in response.retrieved_chunks],
            'avg_retrieval_score': sum(chunk.score for chunk in response.retrieved_chunks) / len(response.retrieved_chunks) if response.retrieved_chunks else 0,
            'chunking_strategies': list(set(chunk.metadata.get('chunking_strategy', 'unknown') for chunk in response.retrieved_chunks)),
            'sources': list(set(chunk.metadata.get('source_url', '') for chunk in response.retrieved_chunks)),
            'answer': response.answer,
            'citations': response.citations
        }
        
        return evaluation
    
    def batch_evaluate(self, queries: List[str]) -> Dict[str, Any]:
        """Evaluate multiple queries."""
        results = []
        
        for query in queries:
            evaluation = self.evaluate_query(query)
            results.append(evaluation)
        
        # Calculate aggregate metrics
        total_time = sum(r['response_time'] for r in results)
        avg_time = total_time / len(results) if results else 0
        avg_chunks = sum(r['chunks_retrieved'] for r in results) / len(results) if results else 0
        avg_score = sum(r['avg_retrieval_score'] for r in results) / len(results) if results else 0
        
        return {
            'total_queries': len(queries),
            'avg_response_time': avg_time,
            'avg_chunks_retrieved': avg_chunks,
            'avg_retrieval_score': avg_score,
            'results': results
        }


def main():
    """Main RAG pipeline testing function."""
    settings = get_settings()
    
    logger.info("🔄 Initializing RAG pipeline...")
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    if not rag_pipeline.llm_model:
        logger.error("❌ Gemini Pro not available. Please check your API key.")
        return
    
    logger.info("🔄 Testing RAG pipeline with sample queries...")
    
    # Test queries
    test_queries = [
        "How do I integrate JioPay with my website?",
        "What are the pricing plans for JioPay Business?",
        "What security features does JioPay offer?",
        "How do I contact JioPay customer support?",
        "What payment methods does JioPay support?",
        "How do I handle refunds with JioPay?",
        "What are the transaction limits?",
        "How do I set up a JioPay merchant account?"
    ]
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_pipeline)
    
    # Evaluate queries
    evaluation_results = evaluator.batch_evaluate(test_queries)
    
    # Save results
    results_file = Path(settings.evaluation_data_dir) / f"rag_evaluation_{get_timestamp()}.json"
    Path(settings.evaluation_data_dir).mkdir(parents=True, exist_ok=True)
    save_json(evaluation_results, str(results_file))
    
    # Print results
    print("\n" + "="*80)
    print("📊 RAG PIPELINE EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n✅ Pipeline Status:")
    print(f"  - Gemini Pro: {'✅ Available' if rag_pipeline.llm_model else '❌ Not Available'}")
    print(f"  - Embedding Model: {'✅ Available' if rag_pipeline.embedding_model else '❌ Not Available'}")
    print(f"  - Vector Collections: {len(rag_pipeline.vector_manager.collections)}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"  - Total Queries: {evaluation_results['total_queries']}")
    print(f"  - Average Response Time: {evaluation_results['avg_response_time']:.2f}s")
    print(f"  - Average Chunks Retrieved: {evaluation_results['avg_chunks_retrieved']:.1f}")
    print(f"  - Average Retrieval Score: {evaluation_results['avg_retrieval_score']:.3f}")
    
    print(f"\n🔍 Sample Responses:")
    for i, result in enumerate(evaluation_results['results'][:3]):  # Show first 3
        print(f"\n{i+1}. Q: {result['query']}")
        print(f"   A: {result['answer'][:150]}...")
        print(f"   Sources: {len(result['sources'])} unique, Score: {result['avg_retrieval_score']:.3f}")
    
    print(f"\n💾 Evaluation results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
