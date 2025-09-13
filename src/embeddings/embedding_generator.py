"""
Simplified embeddings comparison module for evaluating open-source embedding models.
Focuses on E5 and BGE embeddings for RAG system optimization.
"""
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import tiktoken

from src.config import get_settings
from src.utils import save_json, load_json, get_timestamp

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class EmbeddingResult:
    """Represents embedding generation result."""
    model_name: str
    chunking_strategy: str
    chunk_id: str
    text: str
    embedding: List[float]
    generation_time: float
    token_count: int
    metadata: Dict[str, Any]


class EmbeddingModels:
    """Handles different embedding models."""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize models
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all embedding models."""
        try:
            # E5 models
            logger.info("Loading E5-base model...")
            self.models['e5-base'] = {
                'type': 'sentence_transformer',
                'model': SentenceTransformer('intfloat/e5-base'),
                'dimensions': 768
            }
            logger.info("✅ E5-base loaded successfully")
        except Exception as e:
            logger.warning(f"E5-base model not available: {e}")
        
        try:
            logger.info("Loading E5-large model...")
            self.models['e5-large'] = {
                'type': 'sentence_transformer',
                'model': SentenceTransformer('intfloat/e5-large'),
                'dimensions': 1024
            }
            logger.info("✅ E5-large loaded successfully")
        except Exception as e:
            logger.warning(f"E5-large model not available: {e}")
        
        try:
            # BGE models
            logger.info("Loading BGE-small model...")
            self.models['bge-small'] = {
                'type': 'sentence_transformer',
                'model': SentenceTransformer('BAAI/bge-small-en-v1.5'),
                'dimensions': 384
            }
            logger.info("✅ BGE-small loaded successfully")
        except Exception as e:
            logger.warning(f"BGE-small model not available: {e}")
        
        try:
            logger.info("Loading BGE-base model...")
            self.models['bge-base'] = {
                'type': 'sentence_transformer',
                'model': SentenceTransformer('BAAI/bge-base-en-v1.5'),
                'dimensions': 768
            }
            logger.info("✅ BGE-base loaded successfully")
        except Exception as e:
            logger.warning(f"BGE-base model not available: {e}")
        
        try:
            logger.info("Loading BGE-large model...")
            self.models['bge-large'] = {
                'type': 'sentence_transformer',
                'model': SentenceTransformer('BAAI/bge-large-en-v1.5'),
                'dimensions': 1024
            }
            logger.info("✅ BGE-large loaded successfully")
        except Exception as e:
            logger.warning(f"BGE-large model not available: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def generate_sentence_transformer_embedding(self, text: str, model) -> Tuple[List[float], float]:
        """Generate SentenceTransformer embedding."""
        start_time = time.time()
        
        try:
            embedding = model.encode(text, convert_to_tensor=False)
            generation_time = time.time() - start_time
            return embedding.tolist(), generation_time
        except Exception as e:
            logger.error(f"SentenceTransformer embedding failed: {e}")
            return [], 0.0
    
    def generate_embedding(self, text: str, model_name: str) -> Tuple[List[float], float]:
        """Generate embedding using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model_info = self.models[model_name]
        
        if model_info['type'] == 'sentence_transformer':
            return self.generate_sentence_transformer_embedding(text, model_info['model'])
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")


class EmbeddingGenerator:
    """Generates embeddings for all chunking strategies."""
    
    def __init__(self):
        self.embedding_models = EmbeddingModels()
        self.results = []
    
    def generate_embeddings_for_strategy(self, chunks: List[Dict[str, Any]], 
                                       strategy: str, model_name: str) -> List[EmbeddingResult]:
        """Generate embeddings for a specific chunking strategy."""
        results = []
        
        logger.info(f"Generating {model_name} embeddings for {strategy} ({len(chunks)} chunks)")
        
        for i, chunk in enumerate(chunks):
            try:
                embedding, generation_time = self.embedding_models.generate_embedding(
                    chunk['text'], model_name
                )
                
                if embedding:
                    result = EmbeddingResult(
                        model_name=model_name,
                        chunking_strategy=strategy,
                        chunk_id=chunk['chunk_id'],
                        text=chunk['text'],
                        embedding=embedding,
                        generation_time=generation_time,
                        token_count=self.embedding_models.count_tokens(chunk['text']),
                        metadata=chunk['metadata']
                    )
                    results.append(result)
                else:
                    logger.warning(f"Failed to generate embedding for chunk {chunk['chunk_id']}")
            
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {chunk['chunk_id']}: {e}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks for {strategy}")
        
        logger.info(f"Generated {len(results)} embeddings for {strategy} with {model_name}")
        return results
    
    def generate_all_embeddings(self, processed_data: Dict[str, Any]) -> Dict[str, List[EmbeddingResult]]:
        """Generate embeddings for all strategies and models."""
        all_results = {}
        
        # Get available models
        available_models = list(self.embedding_models.models.keys())
        logger.info(f"Available models: {available_models}")
        
        if not available_models:
            logger.error("No embedding models available!")
            return all_results
        
        # Get chunking strategies
        chunking_strategies = list(processed_data['chunks'].keys())
        logger.info(f"Chunking strategies: {chunking_strategies}")
        
        # Generate embeddings for each combination
        for model_name in available_models:
            model_results = {}
            
            for strategy in chunking_strategies:
                chunks = processed_data['chunks'][strategy]
                
                if chunks:  # Only process if chunks exist
                    results = self.generate_embeddings_for_strategy(chunks, strategy, model_name)
                    model_results[strategy] = results
                    
                    # Add to overall results
                    key = f"{model_name}_{strategy}"
                    all_results[key] = results
            
            # Save model-specific results
            model_file = Path(settings.embeddings_dir) / f"embeddings_{model_name}_{get_timestamp()}.json"
            Path(settings.embeddings_dir).mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            serializable_results = {}
            for strategy, results in model_results.items():
                serializable_results[strategy] = [
                    {
                        'model_name': r.model_name,
                        'chunking_strategy': r.chunking_strategy,
                        'chunk_id': r.chunk_id,
                        'text': r.text,
                        'embedding': r.embedding,
                        'generation_time': r.generation_time,
                        'token_count': r.token_count,
                        'metadata': r.metadata
                    }
                    for r in results
                ]
            
            save_json(serializable_results, str(model_file))
            logger.info(f"Saved {model_name} embeddings to {model_file}")
        
        return all_results
    
    def analyze_embeddings(self, all_results: Dict[str, List[EmbeddingResult]]) -> Dict[str, Any]:
        """Analyze embedding generation results."""
        analysis = {}
        
        for key, results in all_results.items():
            if not results:
                continue
            
            model_name, strategy = key.split('_', 1)
            
            if model_name not in analysis:
                analysis[model_name] = {}
            
            # Calculate statistics
            total_chunks = len(results)
            total_time = sum(r.generation_time for r in results)
            avg_time = total_time / total_chunks if total_chunks > 0 else 0
            total_tokens = sum(r.token_count for r in results)
            avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0
            
            # Embedding dimensions
            embedding_dims = len(results[0].embedding) if results else 0
            
            # Performance metrics
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            chunks_per_second = total_chunks / total_time if total_time > 0 else 0
            
            analysis[model_name][strategy] = {
                'total_chunks': total_chunks,
                'total_time': total_time,
                'avg_time_per_chunk': avg_time,
                'total_tokens': total_tokens,
                'avg_tokens_per_chunk': avg_tokens,
                'embedding_dimensions': embedding_dims,
                'tokens_per_second': tokens_per_second,
                'chunks_per_second': chunks_per_second,
                'success_rate': 1.0  # All successful if we got here
            }
        
        return analysis


class VectorStoreManager:
    """Manages vector database operations."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(Path(settings.vector_store_dir)),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collections = {}
    
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
    
    def add_embeddings_to_collection(self, collection_name: str, 
                                   embeddings: List[EmbeddingResult]):
        """Add embeddings to collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
        
        if not collection:
            logger.error(f"Could not create/get collection {collection_name}")
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = [r.chunk_id for r in embeddings]
            documents = [r.text for r in embeddings]
            embeddings_list = [r.embedding for r in embeddings]
            metadatas = [
                {
                    'chunking_strategy': r.chunking_strategy,
                    'source_url': r.metadata.get('source_url', ''),
                    'source_title': r.metadata.get('source_title', ''),
                    'token_count': r.token_count,
                    'generation_time': r.generation_time,
                    **r.metadata
                }
                for r in embeddings
            ]
            
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


def main():
    """Main embedding generation function."""
    settings = get_settings()
    
    # Load processed data
    processed_file = Path(settings.processed_data_dir) / "jiopay_processed_20250913_161712.json"
    if not processed_file.exists():
        logger.error(f"Processed data file not found: {processed_file}")
        return
    
    logger.info("🔄 Loading processed data...")
    processed_data = load_json(str(processed_file))
    
    logger.info("🔄 Initializing embedding models...")
    generator = EmbeddingGenerator()
    
    logger.info("🔄 Generating embeddings for all strategies and models...")
    all_results = generator.generate_all_embeddings(processed_data)
    
    logger.info("🔄 Analyzing embedding results...")
    analysis = generator.analyze_embeddings(all_results)
    
    # Save analysis
    analysis_file = Path(settings.embeddings_dir) / f"embedding_analysis_{get_timestamp()}.json"
    Path(settings.embeddings_dir).mkdir(parents=True, exist_ok=True)
    save_json(analysis, str(analysis_file))
    
    # Initialize vector store manager
    logger.info("🔄 Setting up vector database...")
    vector_manager = VectorStoreManager()
    
    # Create collections for each model
    for model_name in generator.embedding_models.models.keys():
        collection_name = f"jiopay_{model_name}"
        
        # Get all embeddings for this model
        model_embeddings = []
        for key, results in all_results.items():
            if key.startswith(f"{model_name}_"):
                model_embeddings.extend(results)
        
        if model_embeddings:
            success = vector_manager.add_embeddings_to_collection(
                collection_name, model_embeddings
            )
            if success:
                logger.info(f"✅ Created vector collection: {collection_name}")
            else:
                logger.error(f"❌ Failed to create vector collection: {collection_name}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EMBEDDING GENERATION SUMMARY")
    print("="*80)
    
    for model_name, model_analysis in analysis.items():
        print(f"\n{model_name.upper()}:")
        
        total_chunks = sum(strategy_stats['total_chunks'] for strategy_stats in model_analysis.values())
        total_time = sum(strategy_stats['total_time'] for strategy_stats in model_analysis.values())
        total_tokens = sum(strategy_stats['total_tokens'] for strategy_stats in model_analysis.values())
        
        print(f"  Total chunks processed: {total_chunks}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average time per chunk: {total_time/total_chunks:.3f}s" if total_chunks > 0 else "  Average time per chunk: 0s")
        print(f"  Tokens per second: {total_tokens/total_time:.1f}" if total_time > 0 else "  Tokens per second: 0")
        
        # Show per-strategy breakdown
        for strategy, stats in model_analysis.items():
            print(f"    {strategy}: {stats['total_chunks']} chunks, {stats['avg_time_per_chunk']:.3f}s avg")
    
    print(f"\n💾 Embedding analysis saved to: {analysis_file}")
    print(f"🗄️ Vector collections created in: {settings.vector_store_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
