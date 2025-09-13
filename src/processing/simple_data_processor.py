"""
Simplified data processing module for cleaning, normalizing, and chunking JioPay content.
Implements multiple chunking strategies for comprehensive RAG system.
"""
import re
import json
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from src.config import get_settings
from src.utils import clean_text, save_json, load_json, get_timestamp

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    source_url: str
    source_title: str
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    metadata: Dict[str, Any]


class TextProcessor:
    """Handles text cleaning and normalization."""
    
    def __init__(self):
        # Basic English stopwords
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'these', 'those', 'or', 'but', 'if', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'could', 'should',
            'would', 'may', 'might', 'must', 'shall', 'do', 'does', 'did'
        }
        
    def clean_and_normalize(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Basic cleaning
        text = clean_text(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common HTML artifacts
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'\b(click here|read more|learn more|see more)\b',
            r'\b(advertisement|ad|sponsored)\b',
            r'\b(loading|please wait|error|404|500)\b',
            r'\b(facebook|twitter|instagram|linkedin)\b',
            r'\b(all rights reserved)\b',
            r'\b(cookie|privacy|terms|conditions|copyright)\b'
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using simple regex."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


class ChunkingStrategies:
    """Implements multiple chunking strategies."""
    
    def __init__(self):
        self.processor = TextProcessor()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def fixed_chunking(self, text: str, chunk_size: int, overlap: int, 
                      source_url: str, source_title: str) -> List[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        start = 0
        chunk_index = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            # Ensure chunk doesn't exceed token limit
            while self.count_tokens(chunk_text) > chunk_size and len(chunk_text.split()) > 1:
                words_list = chunk_text.split()
                chunk_text = ' '.join(words_list[:-1])
            
            if chunk_text.strip():
                chunk = Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{source_url}_{chunk_index}",
                    source_url=source_url,
                    source_title=source_title,
                    chunk_index=chunk_index,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    chunking_strategy="fixed",
                    metadata={
                        "token_count": self.count_tokens(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + chunk_size - overlap, start + 1)
        
        return chunks
    
    def semantic_chunking(self, text: str, source_url: str, source_title: str, 
                         similarity_threshold: float = 0.7) -> List[Chunk]:
        """Simplified semantic chunking based on sentence similarity."""
        chunks = []
        sentences = self.processor.extract_sentences(text)
        
        if not sentences:
            return chunks
        
        # Simple similarity based on word overlap
        current_chunk = []
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            if not current_chunk:
                current_chunk.append(sentence)
            else:
                # Calculate simple word overlap similarity
                prev_words = set(current_chunk[-1].lower().split())
                curr_words = set(sentence.lower().split())
                
                if prev_words and curr_words:
                    overlap = len(prev_words.intersection(curr_words))
                    union = len(prev_words.union(curr_words))
                    similarity = overlap / union if union > 0 else 0
                else:
                    similarity = 0
                
                if similarity >= similarity_threshold:
                    current_chunk.append(sentence)
                else:
                    # Start new chunk
                    chunk_text = ' '.join(current_chunk)
                    if chunk_text.strip():
                        chunk = Chunk(
                            text=chunk_text.strip(),
                            chunk_id=f"{source_url}_semantic_{chunk_index}",
                            source_url=source_url,
                            source_title=source_title,
                            chunk_index=chunk_index,
                            chunk_size=len(chunk_text.split()),
                            chunk_overlap=0,
                            chunking_strategy="semantic",
                            metadata={
                                "token_count": self.count_tokens(chunk_text),
                                "word_count": len(chunk_text.split()),
                                "char_count": len(chunk_text),
                                "sentence_count": len(current_chunk),
                                "similarity_threshold": similarity_threshold
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    current_chunk = [sentence]
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunk = Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{source_url}_semantic_{chunk_index}",
                    source_url=source_url,
                    source_title=source_title,
                    chunk_index=chunk_index,
                    chunk_size=len(chunk_text.split()),
                    chunk_overlap=0,
                    chunking_strategy="semantic",
                    metadata={
                        "token_count": self.count_tokens(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "sentence_count": len(current_chunk),
                        "similarity_threshold": similarity_threshold
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def structural_chunking(self, text: str, source_url: str, source_title: str) -> List[Chunk]:
        """Structural chunking based on headings and structure."""
        chunks = []
        
        # Split by common structural patterns
        patterns = [
            r'\n\s*#{1,6}\s+',  # Markdown headers
            r'\n\s*[A-Z][A-Z\s]+:\s*\n',  # ALL CAPS headers
            r'\n\s*[A-Z][a-z]+:\s*\n',  # Title case headers
            r'\n\s*Q:\s*',  # Questions
            r'\n\s*A:\s*',  # Answers
            r'\n\s*\d+\.\s*',  # Numbered lists
            r'\n\s*[•·▪▫]\s*',  # Bullet points
        ]
        
        # Try to split by structural patterns
        sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(parts)
            sections = new_sections
        
        chunk_index = 0
        for section in sections:
            section = section.strip()
            if len(section) > 50:  # Only include substantial sections
                chunk = Chunk(
                    text=section,
                    chunk_id=f"{source_url}_structural_{chunk_index}",
                    source_url=source_url,
                    source_title=source_title,
                    chunk_index=chunk_index,
                    chunk_size=len(section.split()),
                    chunk_overlap=0,
                    chunking_strategy="structural",
                    metadata={
                        "token_count": self.count_tokens(section),
                        "word_count": len(section.split()),
                        "char_count": len(section),
                        "section_type": "structural"
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def recursive_chunking(self, text: str, source_url: str, source_title: str) -> List[Chunk]:
        """Recursive chunking with hierarchical fallback."""
        chunks = []
        
        # First try structural chunking
        structural_chunks = self.structural_chunking(text, source_url, source_title)
        
        for chunk in structural_chunks:
            # If chunk is too large, recursively chunk it
            if self.count_tokens(chunk.text) > 1024:
                # Use semantic chunking for large structural chunks
                semantic_chunks = self.semantic_chunking(
                    chunk.text, source_url, source_title, 0.6
                )
                chunks.extend(semantic_chunks)
            else:
                chunks.append(chunk)
        
        # If no structural chunks found, use semantic chunking
        if not chunks:
            chunks = self.semantic_chunking(text, source_url, source_title, 0.7)
        
        # If still no chunks, use fixed chunking
        if not chunks:
            chunks = self.fixed_chunking(text, 512, 64, source_url, source_title)
        
        return chunks
    
    def llm_based_chunking(self, text: str, source_url: str, source_title: str) -> List[Chunk]:
        """LLM-based chunking using instruction-aware segmentation."""
        chunks = []
        sentences = self.processor.extract_sentences(text)
        
        if not sentences:
            return chunks
        
        # Group sentences by topic using simple heuristics
        current_topic = []
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            current_topic.append(sentence)
            
            # Check if we should start a new topic
            should_split = False
            
            # Split on question-answer pairs
            if i < len(sentences) - 1:
                next_sentence = sentences[i + 1]
                if (sentence.strip().endswith('?') and 
                    next_sentence.strip().startswith(('A:', 'Answer:', 'Yes', 'No'))):
                    should_split = True
            
            # Split on topic changes (simple keyword-based)
            topic_keywords = {
                'payment': ['payment', 'transaction', 'billing', 'charge'],
                'security': ['security', 'encryption', 'safe', 'protect'],
                'features': ['feature', 'function', 'capability', 'option'],
                'pricing': ['price', 'cost', 'fee', 'plan', 'subscription'],
                'support': ['support', 'help', 'assistance', 'contact']
            }
            
            current_keywords = set()
            for word in sentence.lower().split():
                for topic, keywords in topic_keywords.items():
                    if word in keywords:
                        current_keywords.add(topic)
            
            # If we have a substantial topic chunk, create it
            if (len(current_topic) >= 3 or should_split or 
                len(' '.join(current_topic).split()) > 200):
                
                chunk_text = ' '.join(current_topic)
                if chunk_text.strip():
                    chunk = Chunk(
                        text=chunk_text.strip(),
                        chunk_id=f"{source_url}_llm_{chunk_index}",
                        source_url=source_url,
                        source_title=source_title,
                        chunk_index=chunk_index,
                        chunk_size=len(chunk_text.split()),
                        chunk_overlap=0,
                        chunking_strategy="llm_based",
                        metadata={
                            "token_count": self.count_tokens(chunk_text),
                            "word_count": len(chunk_text.split()),
                            "char_count": len(chunk_text),
                            "sentence_count": len(current_topic),
                            "detected_topics": list(current_keywords),
                            "chunking_method": "rule_based_llm_simulation"
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_topic = []
        
        # Add final topic
        if current_topic:
            chunk_text = ' '.join(current_topic)
            if chunk_text.strip():
                chunk = Chunk(
                    text=chunk_text.strip(),
                    chunk_id=f"{source_url}_llm_{chunk_index}",
                    source_url=source_url,
                    source_title=source_title,
                    chunk_index=chunk_index,
                    chunk_size=len(chunk_text.split()),
                    chunk_overlap=0,
                    chunking_strategy="llm_based",
                    metadata={
                        "token_count": self.count_tokens(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "sentence_count": len(current_topic),
                        "chunking_method": "rule_based_llm_simulation"
                    }
                )
                chunks.append(chunk)
        
        return chunks


class DataProcessor:
    """Main data processing class."""
    
    def __init__(self):
        self.chunking_strategies = ChunkingStrategies()
        self.processor = TextProcessor()
        
    def process_scraped_data(self, scraped_data: List[Dict[str, Any]]) -> Dict[str, List[Chunk]]:
        """Process scraped data and create chunks using multiple strategies."""
        all_chunks = {
            'fixed_256_0': [],
            'fixed_256_64': [],
            'fixed_512_0': [],
            'fixed_512_64': [],
            'fixed_512_128': [],
            'fixed_1024_0': [],
            'fixed_1024_64': [],
            'fixed_1024_128': [],
            'semantic': [],
            'structural': [],
            'recursive': [],
            'llm_based': []
        }
        
        for item in scraped_data:
            url = item['url']
            
            # Get the best content from results
            best_content = None
            best_title = "Unknown"
            
            for method, result in item['results'].items():
                if result and result.get('content', '').strip():
                    best_content = result['content']
                    best_title = result.get('title', 'Unknown')
                    break
            
            if not best_content:
                logger.warning(f"No content found for {url}")
                continue
            
            # Clean the content
            cleaned_content = self.processor.clean_and_normalize(best_content)
            
            if len(cleaned_content) < 50:
                logger.warning(f"Content too short for {url}: {len(cleaned_content)} chars")
                continue
            
            logger.info(f"Processing {url} - {len(cleaned_content)} characters")
            
            # Fixed chunking variations
            fixed_configs = [
                (256, 0, 'fixed_256_0'),
                (256, 64, 'fixed_256_64'),
                (512, 0, 'fixed_512_0'),
                (512, 64, 'fixed_512_64'),
                (512, 128, 'fixed_512_128'),
                (1024, 0, 'fixed_1024_0'),
                (1024, 64, 'fixed_1024_64'),
                (1024, 128, 'fixed_1024_128')
            ]
            
            for chunk_size, overlap, strategy_name in fixed_configs:
                chunks = self.chunking_strategies.fixed_chunking(
                    cleaned_content, chunk_size, overlap, url, best_title
                )
                all_chunks[strategy_name].extend(chunks)
            
            # Semantic chunking
            semantic_chunks = self.chunking_strategies.semantic_chunking(
                cleaned_content, url, best_title, 0.7
            )
            all_chunks['semantic'].extend(semantic_chunks)
            
            # Structural chunking
            structural_chunks = self.chunking_strategies.structural_chunking(
                cleaned_content, url, best_title
            )
            all_chunks['structural'].extend(structural_chunks)
            
            # Recursive chunking
            recursive_chunks = self.chunking_strategies.recursive_chunking(
                cleaned_content, url, best_title
            )
            all_chunks['recursive'].extend(recursive_chunks)
            
            # LLM-based chunking
            llm_chunks = self.chunking_strategies.llm_based_chunking(
                cleaned_content, url, best_title
            )
            all_chunks['llm_based'].extend(llm_chunks)
        
        return all_chunks
    
    def analyze_chunks(self, all_chunks: Dict[str, List[Chunk]]) -> Dict[str, Any]:
        """Analyze chunk statistics."""
        analysis = {}
        
        for strategy, chunks in all_chunks.items():
            if not chunks:
                analysis[strategy] = {
                    'total_chunks': 0,
                    'avg_chunk_size': 0,
                    'avg_token_count': 0,
                    'total_tokens': 0,
                    'unique_sources': 0
                }
                continue
            
            total_chunks = len(chunks)
            avg_chunk_size = sum(len(chunk.text.split()) for chunk in chunks) / total_chunks
            avg_token_count = sum(chunk.metadata['token_count'] for chunk in chunks) / total_chunks
            total_tokens = sum(chunk.metadata['token_count'] for chunk in chunks)
            unique_sources = len(set(chunk.source_url for chunk in chunks))
            
            analysis[strategy] = {
                'total_chunks': total_chunks,
                'avg_chunk_size': avg_chunk_size,
                'avg_token_count': avg_token_count,
                'total_tokens': total_tokens,
                'unique_sources': unique_sources,
                'chunk_size_distribution': {
                    'small': sum(1 for c in chunks if c.metadata['token_count'] < 256),
                    'medium': sum(1 for c in chunks if 256 <= c.metadata['token_count'] < 512),
                    'large': sum(1 for c in chunks if c.metadata['token_count'] >= 512)
                }
            }
        
        return analysis
    
    def save_processed_data(self, all_chunks: Dict[str, List[Chunk]], 
                           analysis: Dict[str, Any], output_file: str):
        """Save processed chunks and analysis."""
        # Convert chunks to serializable format
        serializable_chunks = {}
        for strategy, chunks in all_chunks.items():
            serializable_chunks[strategy] = [
                {
                    'text': chunk.text,
                    'chunk_id': chunk.chunk_id,
                    'source_url': chunk.source_url,
                    'source_title': chunk.source_title,
                    'chunk_index': chunk.chunk_index,
                    'chunk_size': chunk.chunk_size,
                    'chunk_overlap': chunk.chunk_overlap,
                    'chunking_strategy': chunk.chunking_strategy,
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            ]
        
        # Save everything
        output_data = {
            'chunks': serializable_chunks,
            'analysis': analysis,
            'metadata': {
                'processed_at': get_timestamp(),
                'total_strategies': len(all_chunks),
                'total_chunks': sum(len(chunks) for chunks in all_chunks.values())
            }
        }
        
        save_json(output_data, output_file)
        logger.info(f"Processed data saved to {output_file}")


def main():
    """Main processing function."""
    settings = get_settings()
    
    # Load scraped data
    scraped_file = Path(settings.scraped_data_dir) / "jiopay_enhanced_20250913_161400.json"
    if not scraped_file.exists():
        logger.error(f"Scraped data file not found: {scraped_file}")
        return
    
    logger.info("🔄 Loading scraped data...")
    scraped_data = load_json(str(scraped_file))
    
    logger.info("🔄 Processing data with multiple chunking strategies...")
    processor = DataProcessor()
    
    # Process the data
    all_chunks = processor.process_scraped_data(scraped_data['scraped_data'])
    
    # Analyze chunks
    analysis = processor.analyze_chunks(all_chunks)
    
    # Save processed data
    output_file = Path(settings.processed_data_dir) / f"jiopay_processed_{get_timestamp()}.json"
    Path(settings.processed_data_dir).mkdir(parents=True, exist_ok=True)
    
    processor.save_processed_data(all_chunks, analysis, str(output_file))
    
    # Print summary
    print("\n" + "="*80)
    print("📊 DATA PROCESSING SUMMARY")
    print("="*80)
    
    for strategy, stats in analysis.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Average chunk size: {stats['avg_chunk_size']:.1f} words")
        print(f"  Average token count: {stats['avg_token_count']:.1f}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Unique sources: {stats['unique_sources']}")
    
    print(f"\n💾 Processed data saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
