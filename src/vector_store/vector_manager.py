"""
Vector store management module for ChromaDB operations.
Handles collection creation, embedding storage, and retrieval with proper ID management.
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

from src.config import get_settings
from src.utils import save_json, load_json, get_timestamp

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStoreManager:
    """Manages vector database operations with proper ID handling."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(Path(settings.vector_store_dir)),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collections = {}
        self.embedding_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models for query encoding."""
        try:
            self.embedding_models['e5-base'] = SentenceTransformer('intfloat/e5-base')
            logger.info("✅ E5-base model loaded for queries")
        except Exception as e:
            logger.warning(f"E5-base model not available: {e}")
        
        try:
            self.embedding_models['e5-large'] = SentenceTransformer('intfloat/e5-large')
            logger.info("✅ E5-large model loaded for queries")
        except Exception as e:
            logger.warning(f"E5-large model not available: {e}")
        
        try:
            self.embedding_models['bge-small'] = SentenceTransformer('BAAI/bge-small-en-v1.5')
            logger.info("✅ BGE-small model loaded for queries")
        except Exception as e:
            logger.warning(f"BGE-small model not available: {e}")
        
        try:
            self.embedding_models['bge-base'] = SentenceTransformer('BAAI/bge-base-en-v1.5')
            logger.info("✅ BGE-base model loaded for queries")
        except Exception as e:
            logger.warning(f"BGE-base model not available: {e}")
        
        try:
            self.embedding_models['bge-large'] = SentenceTransformer('BAAI/bge-large-en-v1.5')
            logger.info("✅ BGE-large model loaded for queries")
        except Exception as e:
            logger.warning(f"BGE-large model not available: {e}")
    
    def create_collection(self, collection_name: str, embedding_function=None):
        """Create a new collection."""
        try:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            self.collections[collection_name] = collection
            logger.info(f"Created collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return None
    
    def get_collection(self, collection_name: str):
        """Get existing collection."""
        try:
            collection = self.client.get_collection(collection_name)
            self.collections[collection_name] = collection
            return collection
        except Exception as e:
            logger.warning(f"Collection {collection_name} not found: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def add_embeddings_to_collection(self, collection_name: str, 
                                   embeddings: List[Dict[str, Any]], 
                                   model_name: str):
        """Add embeddings to collection with unique IDs."""
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
        
        if not collection:
            logger.error(f"Could not create/get collection {collection_name}")
            return False
        
        try:
            # Prepare data for ChromaDB with unique IDs
            ids = []
            documents = []
            embeddings_list = []
            metadatas = []
            
            for i, embedding_data in enumerate(embeddings):
                # Create unique ID combining model, strategy, and index
                unique_id = f"{model_name}_{embedding_data['chunking_strategy']}_{i}_{uuid.uuid4().hex[:8]}"
                ids.append(unique_id)
                
                documents.append(embedding_data['text'])
                embeddings_list.append(embedding_data['embedding'])
                
                metadata = {
                    'model_name': model_name,
                    'chunking_strategy': embedding_data['chunking_strategy'],
                    'original_chunk_id': embedding_data['chunk_id'],
                    'source_url': embedding_data['metadata'].get('source_url', ''),
                    'source_title': embedding_data['metadata'].get('source_title', ''),
                    'token_count': embedding_data['token_count'],
                    'generation_time': embedding_data['generation_time'],
                    **embedding_data['metadata']
                }
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to {collection_name}: {e}")
            return False
    
    def query_collection(self, collection_name: str, query_text: str, 
                        n_results: int = 5, model_name: str = None) -> List[RetrievalResult]:
        """Query collection for similar documents."""
        collection = self.get_collection(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return []
        
        try:
            # If model_name is specified, encode query with that model
            if model_name and model_name in self.embedding_models:
                query_embedding = self.embedding_models[model_name].encode(query_text)
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results
                )
            else:
                # Use ChromaDB's default embedding
                results = collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    retrieval_result = RetrievalResult(
                        chunk_id=results['ids'][0][i],
                        text=doc,
                        score=results['distances'][0][i] if results['distances'] else 0.0,
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                    retrieval_results.append(retrieval_result)
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            return {}
        
        try:
            count = collection.count()
            return {
                'name': collection_name,
                'count': count,
                'status': 'active'
            }
        except Exception as e:
            logger.error(f"Error getting stats for {collection_name}: {e}")
            return {}


class RetrievalSystem:
    """Comprehensive retrieval system for RAG."""
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.collections_created = []
    
    def setup_vector_collections(self, embeddings_dir: str):
        """Set up vector collections from embedding files."""
        embeddings_path = Path(embeddings_dir)
        
        if not embeddings_path.exists():
            logger.error(f"Embeddings directory not found: {embeddings_dir}")
            return False
        
        # Find all embedding files
        embedding_files = list(embeddings_path.glob("embeddings_*.json"))
        
        if not embedding_files:
            logger.error("No embedding files found")
            return False
        
        logger.info(f"Found {len(embedding_files)} embedding files")
        
        # Process each embedding file
        for embedding_file in embedding_files:
            try:
                logger.info(f"Processing {embedding_file.name}")
                embeddings_data = load_json(str(embedding_file))
                
                # Extract model name from filename
                model_name = embedding_file.stem.replace('embeddings_', '').split('_')[0]
                collection_name = f"jiopay_{model_name}"
                
                # Delete existing collection if it exists
                self.vector_manager.delete_collection(collection_name)
                
                # Process each chunking strategy
                total_embeddings = 0
                for strategy, embeddings in embeddings_data.items():
                    if embeddings:
                        success = self.vector_manager.add_embeddings_to_collection(
                            collection_name, embeddings, model_name
                        )
                        if success:
                            total_embeddings += len(embeddings)
                            logger.info(f"Added {len(embeddings)} embeddings for {strategy}")
                
                if total_embeddings > 0:
                    self.collections_created.append(collection_name)
                    logger.info(f"✅ Created collection {collection_name} with {total_embeddings} embeddings")
                else:
                    logger.warning(f"No embeddings added to {collection_name}")
                
            except Exception as e:
                logger.error(f"Error processing {embedding_file}: {e}")
        
        return len(self.collections_created) > 0
    
    def test_retrieval(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Test retrieval system with sample queries."""
        if not test_queries:
            test_queries = [
                "How do I make a payment with JioPay?",
                "What are the security features?",
                "How much does JioPay cost?",
                "How do I contact customer support?",
                "What payment methods are accepted?"
            ]
        
        results = {}
        
        for collection_name in self.collections_created:
            model_name = collection_name.replace('jiopay_', '')
            results[model_name] = {}
            
            logger.info(f"Testing retrieval with {model_name}")
            
            for query in test_queries:
                retrieval_results = self.vector_manager.query_collection(
                    collection_name, query, n_results=3, model_name=model_name
                )
                
                results[model_name][query] = [
                    {
                        'chunk_id': r.chunk_id,
                        'text': r.text[:200] + "..." if len(r.text) > 200 else r.text,
                        'score': r.score,
                        'strategy': r.metadata.get('chunking_strategy', 'unknown'),
                        'source': r.metadata.get('source_url', 'unknown')
                    }
                    for r in retrieval_results
                ]
        
        return results
    
    def compare_models(self, query: str, n_results: int = 5) -> Dict[str, List[RetrievalResult]]:
        """Compare retrieval results across different models."""
        results = {}
        
        for collection_name in self.collections_created:
            model_name = collection_name.replace('jiopay_', '')
            retrieval_results = self.vector_manager.query_collection(
                collection_name, query, n_results=n_results, model_name=model_name
            )
            results[model_name] = retrieval_results
        
        return results
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of all collections."""
        summary = {}
        
        for collection_name in self.collections_created:
            stats = self.vector_manager.get_collection_stats(collection_name)
            summary[collection_name] = stats
        
        return summary


def main():
    """Main vector store setup function."""
    settings = get_settings()
    
    logger.info("🔄 Setting up vector store collections...")
    
    # Initialize retrieval system
    retrieval_system = RetrievalSystem()
    
    # Set up collections from embedding files
    success = retrieval_system.setup_vector_collections(settings.embeddings_dir)
    
    if not success:
        logger.error("Failed to set up vector collections")
        return
    
    logger.info("🔄 Testing retrieval system...")
    
    # Test retrieval with sample queries
    test_results = retrieval_system.test_retrieval()
    
    # Save test results
    test_file = Path(settings.vector_store_dir) / f"retrieval_test_{get_timestamp()}.json"
    Path(settings.vector_store_dir).mkdir(parents=True, exist_ok=True)
    save_json(test_results, str(test_file))
    
    # Get collection summary
    summary = retrieval_system.get_collection_summary()
    
    # Save summary
    summary_file = Path(settings.vector_store_dir) / f"collections_summary_{get_timestamp()}.json"
    save_json(summary, str(summary_file))
    
    # Print results
    print("\n" + "="*80)
    print("📊 VECTOR STORE SETUP SUMMARY")
    print("="*80)
    
    print(f"\n✅ Collections Created: {len(retrieval_system.collections_created)}")
    for collection in retrieval_system.collections_created:
        stats = summary.get(collection, {})
        print(f"  - {collection}: {stats.get('count', 0)} embeddings")
    
    print(f"\n🔍 Retrieval Test Results:")
    for model, queries in test_results.items():
        print(f"\n{model.upper()}:")
        for query, results in queries.items():
            print(f"  Q: {query}")
            for i, result in enumerate(results[:2]):  # Show top 2 results
                print(f"    {i+1}. [{result['strategy']}] {result['text']}")
                print(f"       Score: {result['score']:.3f}, Source: {result['source']}")
    
    print(f"\n💾 Test results saved to: {test_file}")
    print(f"📊 Collection summary saved to: {summary_file}")
    print(f"🗄️ Vector store location: {settings.vector_store_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
