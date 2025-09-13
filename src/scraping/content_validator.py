"""
Data validation and quality assessment for scraped JioPay content.
"""
import re
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContentQuality:
    """Content quality metrics."""
    url: str
    title_length: int
    content_length: int
    has_title: bool
    has_content: bool
    noise_ratio: float
    readability_score: float
    language_score: float
    is_valid: bool
    issues: List[str]


class ContentValidator:
    """Validates and assesses quality of scraped content."""
    
    def __init__(self):
        self.min_content_length = 100
        self.max_noise_ratio = 0.3
        self.min_readability_score = 0.5
        
    def validate_content(self, scraped_data: List[Dict[str, Any]]) -> List[ContentQuality]:
        """Validate all scraped content."""
        quality_results = []
        
        for item in scraped_data:
            url = item['url']
            results = item['results']
            
            # Get the best result (first successful method)
            best_result = None
            for method, result in results.items():
                if result and result['content'].strip():
                    best_result = result
                    break
            
            if best_result:
                quality = self._assess_quality(best_result)
                quality_results.append(quality)
            else:
                # No valid content found
                quality_results.append(ContentQuality(
                    url=url,
                    title_length=0,
                    content_length=0,
                    has_title=False,
                    has_content=False,
                    noise_ratio=1.0,
                    readability_score=0.0,
                    language_score=0.0,
                    is_valid=False,
                    issues=["No valid content found"]
                ))
        
        return quality_results
    
    def _assess_quality(self, content_data: Dict[str, Any]) -> ContentQuality:
        """Assess quality of a single content item."""
        url = content_data['url']
        title = content_data.get('title', '')
        content = content_data.get('content', '')
        
        issues = []
        
        # Basic checks
        has_title = bool(title.strip())
        has_content = bool(content.strip())
        
        title_length = len(title)
        content_length = len(content)
        
        if not has_title:
            issues.append("Missing title")
        
        if content_length < self.min_content_length:
            issues.append(f"Content too short ({content_length} chars)")
        
        # Calculate noise ratio
        noise_ratio = self._calculate_noise_ratio(content)
        if noise_ratio > self.max_noise_ratio:
            issues.append(f"High noise ratio ({noise_ratio:.2f})")
        
        # Calculate readability score
        readability_score = self._calculate_readability(content)
        if readability_score < self.min_readability_score:
            issues.append(f"Low readability score ({readability_score:.2f})")
        
        # Calculate language score (English content)
        language_score = self._calculate_language_score(content)
        
        # Overall validity
        is_valid = (
            has_title and 
            has_content and 
            content_length >= self.min_content_length and
            noise_ratio <= self.max_noise_ratio and
            readability_score >= self.min_readability_score
        )
        
        return ContentQuality(
            url=url,
            title_length=title_length,
            content_length=content_length,
            has_title=has_title,
            has_content=has_content,
            noise_ratio=noise_ratio,
            readability_score=readability_score,
            language_score=language_score,
            is_valid=is_valid,
            issues=issues
        )
    
    def _calculate_noise_ratio(self, content: str) -> float:
        """Calculate noise ratio in content."""
        if not content:
            return 1.0
        
        # Common noise patterns
        noise_patterns = [
            r'\b(cookie|privacy|terms|conditions|copyright)\b',
            r'\b(click here|read more|learn more|see more)\b',
            r'\b(advertisement|ad|sponsored)\b',
            r'\b(loading|please wait|error|404|500)\b',
            r'\b(login|sign in|register|sign up)\b',
            r'\b(facebook|twitter|instagram|linkedin)\b',
            r'\b(©|®|™)\b',
            r'\b(all rights reserved)\b'
        ]
        
        noise_count = 0
        total_words = len(content.split())
        
        for pattern in noise_patterns:
            matches = re.findall(pattern, content.lower())
            noise_count += len(matches)
        
        return noise_count / max(total_words, 1)
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate simple readability score."""
        if not content:
            return 0.0
        
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (higher is better)
        readability = 1.0 - (avg_sentence_length / 50.0) - (avg_word_length / 20.0)
        return max(0.0, min(1.0, readability))
    
    def _calculate_language_score(self, content: str) -> float:
        """Calculate English language score."""
        if not content:
            return 0.0
        
        # Common English words
        english_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
            'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us', 'is', 'was', 'are', 'were', 'been',
            'has', 'had', 'having', 'does', 'did', 'doing', 'am', 'being'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        if not words:
            return 0.0
        
        english_word_count = sum(1 for word in words if word in english_words)
        return english_word_count / len(words)
    
    def generate_quality_report(self, quality_results: List[ContentQuality]) -> Dict[str, Any]:
        """Generate quality assessment report."""
        total_items = len(quality_results)
        valid_items = sum(1 for q in quality_results if q.is_valid)
        
        if total_items == 0:
            return {
                'summary': {
                    'total_items': 0,
                    'valid_items': 0,
                    'invalid_items': 0,
                    'validity_rate': 0
                },
                'metrics': {
                    'avg_content_length': 0,
                    'avg_noise_ratio': 0,
                    'avg_readability_score': 0,
                    'avg_language_score': 0
                },
                'common_issues': {},
                'quality_distribution': {
                    'excellent': 0,
                    'good': 0,
                    'fair': 0,
                    'poor': 0
                }
            }
        
        avg_content_length = sum(q.content_length for q in quality_results) / total_items
        avg_noise_ratio = sum(q.noise_ratio for q in quality_results) / total_items
        avg_readability = sum(q.readability_score for q in quality_results) / total_items
        avg_language_score = sum(q.language_score for q in quality_results) / total_items
        
        # Common issues
        issue_counts = {}
        for quality in quality_results:
            for issue in quality.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'summary': {
                'total_items': total_items,
                'valid_items': valid_items,
                'invalid_items': total_items - valid_items,
                'validity_rate': valid_items / total_items if total_items > 0 else 0
            },
            'metrics': {
                'avg_content_length': avg_content_length,
                'avg_noise_ratio': avg_noise_ratio,
                'avg_readability_score': avg_readability,
                'avg_language_score': avg_language_score
            },
            'common_issues': issue_counts,
            'quality_distribution': {
                'excellent': sum(1 for q in quality_results if q.readability_score > 0.8),
                'good': sum(1 for q in quality_results if 0.6 < q.readability_score <= 0.8),
                'fair': sum(1 for q in quality_results if 0.4 < q.readability_score <= 0.6),
                'poor': sum(1 for q in quality_results if q.readability_score <= 0.4)
            }
        }
