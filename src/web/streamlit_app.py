"""
Simplified web interface for JioPay RAG Chatbot using Streamlit.
Handles memory constraints and import issues.
"""
import streamlit as st
import json
import time
import sys
import os
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables to handle memory issues
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

try:
    from src.rag.simple_rag_pipeline import RAGPipeline, RAGResponse
    from src.config import get_settings
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="JioPay RAG Chatbot",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def initialize_rag_pipeline():
    """Initialize the RAG pipeline with memory management."""
    if st.session_state.rag_pipeline is None:
        with st.spinner("Initializing RAG pipeline (this may take a moment)..."):
            try:
                st.session_state.rag_pipeline = RAGPipeline()
                return True
            except Exception as e:
                st.error(f"Failed to initialize RAG pipeline: {e}")
                return False
    return True


def display_citations(citations: List[Dict[str, Any]]):
    """Display citations in an expandable section."""
    if not citations:
        return
    
    with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
        for i, citation in enumerate(citations, 1):
            st.markdown(f"**Source {i}:**")
            st.markdown(f"- **Text:** {citation['text'][:200]}...")
            st.markdown(f"- **Score:** {citation['score']:.3f}")
            st.markdown(f"- **Strategy:** {citation['chunking_strategy']}")
            st.markdown(f"- **Model:** {citation['model_name']}")
            if citation['source_url']:
                st.markdown(f"- **URL:** {citation['source_url']}")
            st.markdown("---")


def display_response_metrics(response: RAGResponse):
    """Display response metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Response Time", f"{response.response_time:.2f}s")
    
    with col2:
        st.metric("Chunks Retrieved", len(response.retrieved_chunks))
    
    with col3:
        avg_score = sum(chunk.score for chunk in response.retrieved_chunks) / len(response.retrieved_chunks) if response.retrieved_chunks else 0
        st.metric("Avg Retrieval Score", f"{avg_score:.3f}")
    
    with col4:
        st.metric("Model Used", response.retrieval_model)


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("💳 JioPay RAG Chatbot")
    st.markdown("Ask questions about JioPay's features, pricing, security, and support!")
    
    # Initialize pipeline
    if not initialize_rag_pipeline():
        st.stop()
    
    rag_pipeline = st.session_state.rag_pipeline
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection - only show available models
        available_collections = []
        for collection_name in ["jiopay_bge-small", "jiopay_bge-base", "jiopay_bge-large", "jiopay_e5-base", "jiopay_e5-large"]:
            try:
                collection = rag_pipeline.vector_manager.get_collection(collection_name)
                if collection:
                    available_collections.append(collection_name)
            except:
                pass
        
        if not available_collections:
            st.error("No vector collections available. Please check your setup.")
            st.stop()
        
        collection_options = {
            "BGE-Small (Fast)": "jiopay_bge-small",
            "BGE-Base (Balanced)": "jiopay_bge-base", 
            "BGE-Large (Quality)": "jiopay_bge-large",
            "E5-Base": "jiopay_e5-base",
            "E5-Large": "jiopay_e5-large"
        }
        
        # Filter options based on available collections
        filtered_options = {k: v for k, v in collection_options.items() if v in available_collections}
        
        selected_model = st.selectbox(
            "Retrieval Model",
            options=list(filtered_options.keys()),
            index=0
        )
        
        # Number of results
        n_results = st.slider("Number of Results", 1, 10, 5)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_conversation()
            st.rerun()
        
        # Pipeline status
        st.header("📊 Pipeline Status")
        st.success("✅ RAG Pipeline Ready")
        st.info(f"LLM: {'✅ Available' if rag_pipeline.llm_model else '❌ Mock Mode'}")
        st.info(f"Embeddings: {'✅ Available' if rag_pipeline.embedding_model else '❌ Not Available'}")
        st.info(f"Available Collections: {len(available_collections)}")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if available
            if message["role"] == "assistant" and "citations" in message:
                display_citations(message["citations"])
            
            # Display metrics if available
            if message["role"] == "assistant" and "metrics" in message:
                display_response_metrics(message["metrics"])
    
    # Chat input
    if prompt := st.chat_input("Ask about JioPay..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    collection_name = filtered_options[selected_model]
                    response = rag_pipeline.process_query(
                        prompt, 
                        collection_name=collection_name,
                        n_results=n_results
                    )
                    
                    # Display answer
                    st.markdown(response.answer)
                    
                    # Display citations
                    display_citations(response.citations)
                    
                    # Display metrics
                    display_response_metrics(response)
                    
                    # Add assistant message to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.answer,
                        "citations": response.citations,
                        "metrics": response
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
    
    # Sample questions
    st.header("💡 Sample Questions")
    
    sample_questions = [
        "How do I integrate JioPay with my website?",
        "What are the pricing plans for JioPay Business?",
        "What security features does JioPay offer?",
        "How do I contact JioPay customer support?",
        "What payment methods does JioPay support?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"sample_{i}"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            collection_name = filtered_options[selected_model]
                            response = rag_pipeline.process_query(
                                question, 
                                collection_name=collection_name,
                                n_results=n_results
                            )
                            
                            # Display answer
                            st.markdown(response.answer)
                            
                            # Display citations
                            display_citations(response.citations)
                            
                            # Display metrics
                            display_response_metrics(response)
                            
                            # Add assistant message to session state
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response.answer,
                                "citations": response.citations,
                                "metrics": response
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing query: {e}")
                
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**JioPay RAG Chatbot** - Powered by Retrieval-Augmented Generation")
    st.markdown("Built with Streamlit, ChromaDB, and Gemini Pro")


if __name__ == "__main__":
    main()