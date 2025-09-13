"""
Utility functions and common modules for the JioPay RAG Chatbot.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_text(text: str) -> str:
    """Basic text cleaning function."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove common HTML artifacts
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    
    return text.strip()


def extract_metadata(url: str, title: str = "", content_length: int = 0) -> Dict[str, Any]:
    """Extract metadata from scraped content."""
    return {
        "url": url,
        "title": title,
        "content_length": content_length,
        "scraped_at": get_timestamp(),
        "source": "jiopay_scraper"
    }


def validate_url(url: str) -> bool:
    """Validate if URL is a valid JioPay URL."""
    valid_domains = ["jiopay.com", "www.jiopay.com"]
    return any(domain in url for domain in valid_domains)


def chunk_text_by_size(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Split text into chunks of specified size with overlap."""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (placeholder for now)."""
    # This is a placeholder - will be replaced with proper embedding similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time
        
        logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - {elapsed}")
    
    def finish(self):
        """Mark as finished."""
        elapsed = datetime.now() - self.start_time
        logger.info(f"{self.description} completed in {elapsed}")


def format_citation(citation: Dict[str, Any]) -> str:
    """Format citation for display."""
    url = citation.get("url", "")
    title = citation.get("title", "")
    snippet = citation.get("snippet", "")
    
    if title:
        return f"[{title}]({url})"
    else:
        return f"[Source]({url})"


def format_response_with_citations(answer: str, citations: List[Dict[str, Any]]) -> str:
    """Format response with inline citations."""
    if not citations:
        return answer
    
    formatted_answer = answer
    for i, citation in enumerate(citations, 1):
        citation_text = format_citation(citation)
        formatted_answer += f"\n\n[{i}] {citation_text}"
    
    return formatted_answer
