"""
Main scraping script with comprehensive data collection and validation.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any

from src.scraping.jiopay_scraper import JioPayScraper
from src.scraping.content_validator import ContentValidator
from src.config import get_settings
from src.utils import save_json, get_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_comprehensive_scraping() -> Dict[str, Any]:
    """Run comprehensive scraping with validation."""
    settings = get_settings()
    
    logger.info("🚀 Starting comprehensive JioPay data collection...")
    
    # Ensure data directory exists
    Path(settings.scraped_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Scrape all content
    logger.info("📡 Phase 1: Scraping JioPay websites...")
    async with JioPayScraper() as scraper:
        scraping_results = await scraper.scrape_all()
    
    # Step 2: Validate content quality
    logger.info("🔍 Phase 2: Validating content quality...")
    validator = ContentValidator()
    quality_results = validator.validate_content(scraping_results['scraped_data'])
    quality_report = validator.generate_quality_report(quality_results)
    
    # Step 3: Prepare final results
    final_results = {
        'scraping_summary': {
            'total_urls': scraping_results['total_urls'],
            'successful_scrapes': scraping_results['successful_scrapes'],
            'failed_scrapes': scraping_results['failed_scrapes'],
            'success_rate': scraping_results['successful_scrapes'] / scraping_results['total_urls'] if scraping_results['total_urls'] > 0 else 0
        },
        'quality_summary': quality_report['summary'],
        'quality_metrics': quality_report['metrics'],
        'common_issues': quality_report['common_issues'],
        'quality_distribution': quality_report['quality_distribution'],
        'scraped_data': scraping_results['scraped_data'],
        'failed_urls': scraping_results['failed_urls'],
        'quality_details': [
            {
                'url': q.url,
                'is_valid': q.is_valid,
                'content_length': q.content_length,
                'noise_ratio': q.noise_ratio,
                'readability_score': q.readability_score,
                'issues': q.issues
            }
            for q in quality_results
        ],
        'metadata': {
            'scraped_at': get_timestamp(),
            'scraper_version': '1.0.0',
            'total_processing_time': time.time()
        }
    }
    
    # Step 4: Save results
    timestamp = get_timestamp()
    output_file = Path(settings.scraped_data_dir) / f"jiopay_comprehensive_{timestamp}.json"
    save_json(final_results, str(output_file))
    
    # Step 5: Print comprehensive report
    print_comprehensive_report(final_results, output_file)
    
    return final_results


def print_comprehensive_report(results: Dict[str, Any], output_file: Path):
    """Print a comprehensive scraping report."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE JIOPAY DATA COLLECTION REPORT")
    print("="*80)
    
    # Scraping Summary
    scraping = results['scraping_summary']
    print(f"\n🔍 SCRAPING SUMMARY:")
    print(f"  Total URLs discovered: {scraping['total_urls']}")
    print(f"  Successfully scraped: {scraping['successful_scrapes']}")
    print(f"  Failed scrapes: {scraping['failed_scrapes']}")
    print(f"  Success rate: {scraping['success_rate']:.1%}")
    
    # Quality Summary
    quality = results['quality_summary']
    print(f"\n✅ QUALITY SUMMARY:")
    print(f"  Valid content items: {quality['valid_items']}")
    print(f"  Invalid content items: {quality['invalid_items']}")
    print(f"  Validity rate: {quality['validity_rate']:.1%}")
    
    # Quality Metrics
    metrics = results['quality_metrics']
    print(f"\n📈 QUALITY METRICS:")
    print(f"  Average content length: {metrics['avg_content_length']:.0f} characters")
    print(f"  Average noise ratio: {metrics['avg_noise_ratio']:.3f}")
    print(f"  Average readability score: {metrics['avg_readability_score']:.3f}")
    print(f"  Average language score: {metrics['avg_language_score']:.3f}")
    
    # Quality Distribution
    distribution = results['quality_distribution']
    print(f"\n📊 CONTENT QUALITY DISTRIBUTION:")
    print(f"  Excellent (readability > 0.8): {distribution['excellent']}")
    print(f"  Good (0.6 < readability ≤ 0.8): {distribution['good']}")
    print(f"  Fair (0.4 < readability ≤ 0.6): {distribution['fair']}")
    print(f"  Poor (readability ≤ 0.4): {distribution['poor']}")
    
    # Common Issues
    issues = results['common_issues']
    if issues:
        print(f"\n⚠️  COMMON ISSUES:")
        for issue, count in sorted(issues.items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue}: {count} occurrences")
    
    # Failed URLs
    failed_urls = results['failed_urls']
    if failed_urls:
        print(f"\n❌ FAILED URLs ({len(failed_urls)}):")
        for url in failed_urls[:10]:  # Show first 10
            print(f"  - {url}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)


async def main():
    """Main function."""
    try:
        results = await run_comprehensive_scraping()
        
        # Additional analysis
        logger.info("📋 Generating additional analysis...")
        
        # Count by scraping method
        method_counts = {}
        for item in results['scraped_data']:
            for method in item['results'].keys():
                method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"\n🔧 SCRAPING METHOD EFFECTIVENESS:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} successful scrapes")
        
        logger.info("✅ Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
