"""
Web scraping module for JioPay business website and help center.
Implements multiple scraping pipelines for comprehensive data collection.
"""
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import trafilatura
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Optional
import time
import logging
from pathlib import Path
import json

from src.config import get_settings
from src.utils import clean_text, extract_metadata, validate_url, save_json, ProgressTracker

logger = logging.getLogger(__name__)
settings = get_settings()


class JioPayScraper:
    """Main scraper class for JioPay websites."""
    
    def __init__(self):
        self.session = None
        self.scraped_data = []
        self.failed_urls = []
        self.base_urls = {
            'main': 'https://jiopay.com',
            'business': 'https://jiopay.com/business',
            'help': 'https://jiopay.com/help',
            'support': 'https://jiopay.com/support',
            'faq': 'https://jiopay.com/faq',
            'about': 'https://jiopay.com/about',
            'features': 'https://jiopay.com/features',
            'pricing': 'https://jiopay.com/pricing',
            'contact': 'https://jiopay.com/contact',
            'terms': 'https://jiopay.com/terms',
            'privacy': 'https://jiopay.com/privacy'
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification for testing
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': settings.user_agent},
            timeout=aiohttp.ClientTimeout(total=30),
            connector=connector
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def discover_urls(self, base_url: str) -> List[str]:
        """Discover URLs from sitemap or by crawling."""
        urls = set()
        
        try:
            # Try to find sitemap
            sitemap_urls = [
                f"{base_url}/sitemap.xml",
                f"{base_url}/sitemap_index.xml",
                f"{base_url}/robots.txt"
            ]
            
            for sitemap_url in sitemap_urls:
                try:
                    response = requests.get(sitemap_url, timeout=10, verify=False)
                    if response.status_code == 200:
                        if 'sitemap' in sitemap_url:
                            urls.update(self._parse_sitemap(response.text, base_url))
                        elif 'robots.txt' in sitemap_url:
                            urls.update(self._parse_robots_txt(response.text, base_url))
                except Exception as e:
                    logger.warning(f"Failed to parse {sitemap_url}: {e}")
            
            # If no sitemap found, crawl the main page for links
            if not urls:
                urls.update(self._crawl_for_links(base_url))
                
        except Exception as e:
            logger.error(f"Error discovering URLs from {base_url}: {e}")
            
        return list(urls)
    
    def _parse_sitemap(self, sitemap_content: str, base_url: str) -> List[str]:
        """Parse XML sitemap to extract URLs."""
        urls = []
        try:
            soup = BeautifulSoup(sitemap_content, 'xml')
            for loc in soup.find_all('loc'):
                url = loc.text.strip()
                if validate_url(url) and 'jiopay.com' in url:
                    urls.append(url)
        except Exception as e:
            logger.warning(f"Error parsing sitemap: {e}")
        return urls
    
    def _parse_robots_txt(self, robots_content: str, base_url: str) -> List[str]:
        """Parse robots.txt to find sitemap URLs."""
        urls = []
        try:
            for line in robots_content.split('\n'):
                if line.strip().startswith('Sitemap:'):
                    sitemap_url = line.split('Sitemap:')[1].strip()
                    if validate_url(sitemap_url):
                        urls.extend(self._parse_sitemap(requests.get(sitemap_url).text, base_url))
        except Exception as e:
            logger.warning(f"Error parsing robots.txt: {e}")
        return urls
    
    def _crawl_for_links(self, base_url: str) -> List[str]:
        """Crawl a page to find relevant links."""
        urls = set()
        try:
            response = requests.get(base_url, timeout=10, verify=False)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    if validate_url(full_url) and 'jiopay.com' in full_url:
                        # Filter for relevant pages
                        if any(keyword in full_url.lower() for keyword in 
                              ['business', 'help', 'support', 'faq', 'guide', 'documentation']):
                            urls.add(full_url)
        except Exception as e:
            logger.warning(f"Error crawling {base_url}: {e}")
        return list(urls)
    
    async def scrape_with_requests(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape using requests + BeautifulSoup."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract content
                    title = soup.find('title')
                    title_text = title.text.strip() if title else ""
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Get main content
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                    if not main_content:
                        main_content = soup.find('body')
                    
                    content = main_content.get_text() if main_content else soup.get_text()
                    content = clean_text(content)
                    
                    return {
                        'url': url,
                        'title': title_text,
                        'content': content,
                        'method': 'requests_bs4',
                        'metadata': extract_metadata(url, title_text, len(content))
                    }
                    
        except Exception as e:
            logger.error(f"Requests scraping failed for {url}: {e}")
            return None
    
    async def scrape_with_trafilatura(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape using trafilatura for better content extraction."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract with trafilatura
                    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
                    
                    if extracted:
                        # Get title separately
                        title = trafilatura.extract(html, include_comments=False, output_format='xml')
                        title_text = ""
                        if title:
                            soup = BeautifulSoup(title, 'xml')
                            title_elem = soup.find('title')
                            if title_elem:
                                title_text = title_elem.text.strip()
                        
                        return {
                            'url': url,
                            'title': title_text,
                            'content': clean_text(extracted),
                            'method': 'trafilatura',
                            'metadata': extract_metadata(url, title_text, len(extracted))
                        }
                    
        except Exception as e:
            logger.error(f"Trafilatura scraping failed for {url}: {e}")
            return None
    
    async def scrape_with_playwright(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape using Playwright for dynamic content."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set viewport and user agent
                await page.set_viewport_size({"width": 1920, "height": 1080})
                await page.set_extra_http_headers({"User-Agent": settings.user_agent})
                
                # Navigate to page
                await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                
                # Wait for content to load - try multiple strategies
                try:
                    # Wait for any content to appear
                    await page.wait_for_selector('body', timeout=10000)
                    
                    # Wait for React/Vue/Angular to load and route to correct page
                    await page.wait_for_timeout(5000)
                    
                    # For SPAs, wait for URL to change or content to update
                    try:
                        # Wait for navigation to complete
                        await page.wait_for_load_state('networkidle', timeout=10000)
                    except:
                        pass
                    
                    # Additional wait for dynamic content
                    await page.wait_for_timeout(2000)
                    
                    # Try to wait for specific content indicators
                    try:
                        await page.wait_for_selector('main, article, .content, [class*="content"], [id*="content"]', timeout=5000)
                    except:
                        pass  # Continue if no specific content found
                    
                except Exception as e:
                    logger.warning(f"Timeout waiting for content on {url}: {e}")
                
                # Extract content with better strategy
                title = await page.title()
                
                # Try multiple content extraction strategies
                content = await page.evaluate("""
                    () => {
                        // Strategy 1: Look for main content areas
                        const contentSelectors = [
                            'main', 'article', '.content', '#content',
                            '[class*="content"]', '[id*="content"]',
                            '.main-content', '.page-content', '.body-content',
                            '.container', '.wrapper'
                        ];
                        
                        let mainElement = null;
                        for (const selector of contentSelectors) {
                            const element = document.querySelector(selector);
                            if (element && element.innerText.trim().length > 100) {
                                mainElement = element;
                                break;
                            }
                        }
                        
                        // Strategy 2: If no main content found, use body but filter better
                        if (!mainElement) {
                            mainElement = document.body;
                        }
                        
                        // Remove unwanted elements
                        const unwanted = mainElement.querySelectorAll(`
                            script, style, nav, footer, header, 
                            .advertisement, .ads, .ad, .sidebar,
                            .menu, .navigation, .nav, .breadcrumb,
                            .social, .share, .comments, .related,
                            [class*="ad"], [id*="ad"], [class*="menu"],
                            [class*="nav"], [class*="sidebar"]
                        `);
                        unwanted.forEach(el => el.remove());
                        
                        // Get text content
                        let text = mainElement.innerText || mainElement.textContent || '';
                        
                        // Clean up the text
                        text = text.replace(/\\s+/g, ' ').trim();
                        
                        // If still too short, try getting all text from body
                        if (text.length < 100) {
                            const bodyText = document.body.innerText || document.body.textContent || '';
                            text = bodyText.replace(/\\s+/g, ' ').trim();
                        }
                        
                        return text;
                    }
                """)
                
                await browser.close()
                
                # Additional content validation
                cleaned_content = clean_text(content)
                
                # Check if we got meaningful content
                if len(cleaned_content) < 50 or "enable javascript" in cleaned_content.lower():
                    logger.warning(f"Playwright got minimal content for {url}: {cleaned_content[:100]}...")
                    return None
                
                return {
                    'url': url,
                    'title': title,
                    'content': cleaned_content,
                    'method': 'playwright',
                    'metadata': extract_metadata(url, title, len(cleaned_content))
                }
                
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {e}")
            return None
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL using multiple methods."""
        results = {}
        
        # Try different scraping methods - prioritize Playwright for SPAs
        methods = [
            ('playwright', self.scrape_with_playwright),
            ('requests_bs4', self.scrape_with_requests),
            ('trafilatura', self.scrape_with_trafilatura)
        ]
        
        for method_name, method_func in methods:
            try:
                result = await method_func(url)
                if result and result['content'].strip() and len(result['content']) > 100:
                    results[method_name] = result
                    logger.info(f"Successfully scraped {url} with {method_name}")
                    break  # Use first successful method with substantial content
            except Exception as e:
                logger.warning(f"Method {method_name} failed for {url}: {e}")
        
        if not results:
            self.failed_urls.append(url)
            logger.error(f"All scraping methods failed for {url}")
        
        return results
    
    async def scrape_all(self) -> Dict[str, Any]:
        """Scrape all JioPay websites."""
        all_urls = set()
        
        # Discover URLs from each base URL
        for site_name, base_url in self.base_urls.items():
            logger.info(f"Discovering URLs for {site_name}: {base_url}")
            urls = self.discover_urls(base_url)
            all_urls.update(urls)
            logger.info(f"Found {len(urls)} URLs for {site_name}")
        
        # Add base URLs if not already included
        all_urls.update(self.base_urls.values())
        
        logger.info(f"Total URLs to scrape: {len(all_urls)}")
        
        # Create progress tracker
        progress = ProgressTracker(len(all_urls), "Scraping JioPay websites")
        
        # Scrape all URLs
        for url in all_urls:
            try:
                result = await self.scrape_url(url)
                if result:
                    self.scraped_data.append({
                        'url': url,
                        'results': result,
                        'scraped_at': time.time()
                    })
                
                progress.update()
                
                # Rate limiting
                await asyncio.sleep(settings.scraping_delay)
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                self.failed_urls.append(url)
                progress.update()
        
        progress.finish()
        
        return {
            'scraped_data': self.scraped_data,
            'failed_urls': self.failed_urls,
            'total_urls': len(all_urls),
            'successful_scrapes': len(self.scraped_data),
            'failed_scrapes': len(self.failed_urls)
        }


async def main():
    """Main function to run the scraper."""
    logger.info("Starting JioPay website scraping...")
    
    # Ensure data directory exists
    Path(settings.scraped_data_dir).mkdir(parents=True, exist_ok=True)
    
    async with JioPayScraper() as scraper:
        results = await scraper.scrape_all()
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = Path(settings.scraped_data_dir) / f"jiopay_scraped_{timestamp}.json"
        save_json(results, str(output_file))
        
        # Print summary
        print(f"\n📊 Scraping Summary:")
        print(f"Total URLs: {results['total_urls']}")
        print(f"Successful: {results['successful_scrapes']}")
        print(f"Failed: {results['failed_scrapes']}")
        print(f"Success Rate: {results['successful_scrapes']/results['total_urls']*100:.1f}%")
        print(f"Results saved to: {output_file}")
        
        if results['failed_urls']:
            print(f"\n❌ Failed URLs:")
            for url in results['failed_urls']:
                print(f"  - {url}")


if __name__ == "__main__":
    asyncio.run(main())
