"""
Intelligent Web Scraping System for Living-LLM.
Continuously learns from web content with quality filtering and processing.
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
import logging
from urllib.parse import urljoin, urlparse, parse_qs
import re
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime, timedelta
import sqlite3
import json
import spacy
from newspaper import Article
import feedparser
import requests_cache
from fake_useragent import UserAgent
from ratelimit import limits, sleep_and_retry


@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations."""
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_second: int = 2
    concurrent_sessions: int = 5
    
    # Content filtering
    min_content_length: int = 500
    max_content_length: int = 50000
    content_quality_threshold: float = 0.7
    
    # Language filtering
    target_languages: List[str] = field(default_factory=lambda: ['en'])
    
    # Source diversity
    max_pages_per_domain: int = 100
    domain_cooldown_hours: int = 24
    
    # Content types
    allowed_content_types: List[str] = field(default_factory=lambda: [
        'text/html', 'application/xhtml+xml'
    ])
    
    # User agents and headers
    rotate_user_agents: bool = True
    use_random_delays: bool = True
    min_delay: float = 1.0
    max_delay: float = 3.0
    
    # Storage
    cache_enabled: bool = True
    cache_duration_hours: int = 24
    database_path: str = "web_learning.db"
    
    # JavaScript rendering
    use_selenium: bool = False
    selenium_timeout: int = 10
    headless_mode: bool = True
    
    # Content sources
    news_feeds: List[str] = field(default_factory=lambda: [
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.reddit.com/r/technology/.rss",
        "https://news.ycombinator.com/rss",
    ])
    
    # Quality metrics
    readability_threshold: float = 30.0  # Flesch reading ease
    information_density_threshold: float = 0.1


@dataclass
class ScrapedContent:
    """Represents scraped web content."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    quality_score: float
    language: str
    content_hash: str
    source_domain: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'quality_score': self.quality_score,
            'language': self.language,
            'content_hash': self.content_hash,
            'source_domain': self.source_domain
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedContent':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ContentQualityAssessor:
    """Assesses the quality of scraped content."""
    
    def __init__(self):
        # Load spaCy model for language processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Quality indicators
        self.quality_patterns = {
            'high_quality': [
                r'\b(research|study|analysis|report|findings)\b',
                r'\b(according to|data shows|statistics indicate)\b',
                r'\b(methodology|experiment|survey|investigation)\b'
            ],
            'low_quality': [
                r'\b(click here|amazing|incredible|shocking)\b',
                r'\b(you won\'t believe|doctors hate|this one trick)\b',
                r'\b(act now|limited time|special offer)\b',
                r'\b(BREAKING|URGENT|MUST READ)\b'
            ]
        }
        
        # Compile patterns
        self.high_quality_regex = re.compile('|'.join(self.quality_patterns['high_quality']), re.IGNORECASE)
        self.low_quality_regex = re.compile('|'.join(self.quality_patterns['low_quality']), re.IGNORECASE)
    
    def assess_quality(self, content: str, title: str = "") -> float:
        """Assess content quality and return score 0-1."""
        if not content or len(content.strip()) < 100:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length score (optimal range 1000-10000 characters)
        length_score = min(1.0, len(content) / 1000) * 0.2
        if len(content) > 10000:
            length_score *= (10000 / len(content))  # Penalty for too long
        score += length_score
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:  # Good sentence length
            score += 0.1
        
        # Vocabulary diversity
        words = re.findall(r'\b\w+\b', content.lower())
        unique_words = set(words)
        if words:
            diversity = len(unique_words) / len(words)
            score += diversity * 0.2
        
        # Quality pattern matching
        high_quality_matches = len(self.high_quality_regex.findall(content + " " + title))
        low_quality_matches = len(self.low_quality_regex.findall(content + " " + title))
        
        
        #score += (high_quality_matches * 0.1) - (low_quality_matches * 0.15) #old line for scoring 
        
        #the new lines for scoring below is supposed to prvent if an article has 50 matches 
        #and an article with 3 matches both end up at 1.0 
        #the pattern matching dominates everything else (length, vocabulary diversity, readability) 
        #and makes those factors meaningless.
        #meaning if one word shows up 50 times it messes with the scoring new code should fix that
        high_quality_bonus = min(0.3, high_quality_matches * 0.1)  # caps at +0.3 max
        low_quality_penalty = min(0.3, low_quality_matches * 0.15)  # caps at -0.3 max
        score += high_quality_bonus - low_quality_penalty
        
        
        # Information density (ratio of informative words)
        if self.nlp:
            doc = self.nlp(content[:1000])  # Sample first 1000 chars
            informative_tokens = [token for token in doc 
                                if not token.is_stop and not token.is_punct 
                                and token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            if len(doc) > 0:
                info_density = len(informative_tokens) / len(doc)
                score += info_density * 0.2
        
        # Readability approximation (simplified Flesch)
        sentences_count = len([s for s in sentences if len(s.strip()) > 5])
        words_count = len(words)
        if sentences_count > 0 and words_count > 0:
            avg_sentence_len = words_count / sentences_count
            if 15 <= avg_sentence_len <= 20:  # Optimal readability
                score += 0.1
        
        return max(0.0, min(1.0, score))


class WebContentDatabase:
    """Database for storing and managing scraped content."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=8000;")
        return conn

    def init_database(self):
        """Initialize the database schema."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS scraped_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT,
                metadata TEXT,
                timestamp TEXT,
                quality_score REAL,
                language TEXT,
                content_hash TEXT UNIQUE,
                source_domain TEXT,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS domain_stats (
                domain TEXT PRIMARY KEY,
                pages_scraped INTEGER DEFAULT 0,
                last_scraped TIMESTAMP,
                avg_quality REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_quality_score ON scraped_content(quality_score);
            CREATE INDEX IF NOT EXISTS idx_timestamp     ON scraped_content(timestamp);
            CREATE INDEX IF NOT EXISTS idx_domain        ON scraped_content(source_domain);
            CREATE INDEX IF NOT EXISTS idx_processed     ON scraped_content(processed);
            """)
            conn.commit()
        finally:
            conn.close()
    
    def store_content(self, content: ScrapedContent) -> bool:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO scraped_content
                (url, title, content, metadata, timestamp, quality_score, language, content_hash, source_domain)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content.url, content.title, content.content,
                json.dumps(content.metadata), content.timestamp.isoformat(),
                content.quality_score, content.language, content.content_hash,
                content.source_domain
            ))

            cursor.execute('''
                INSERT OR REPLACE INTO domain_stats (domain, pages_scraped, last_scraped, avg_quality)
                VALUES (?, 
                    COALESCE((SELECT pages_scraped FROM domain_stats WHERE domain = ?), 0) + 1,
                    ?,
                    (SELECT AVG(quality_score) FROM scraped_content WHERE source_domain = ?)
                )
            ''', (content.source_domain, content.source_domain,
                content.timestamp.isoformat(), content.source_domain))

            conn.commit()
            return True

        except sqlite3.IntegrityError:
            logging.warning(f"Duplicate content detected: {content.url}")
            return False
        except Exception as e:
            logging.error(f"Error storing content: {e}")
            return False
        finally:
            conn.close()
    
    def get_unprocessed_content(self, limit: int = 100) -> List[ScrapedContent]:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT url, title, content, metadata, timestamp, quality_score, language, content_hash, source_domain
                FROM scraped_content
                WHERE processed = FALSE AND quality_score > 0.5
                ORDER BY quality_score DESC, timestamp DESC
                LIMIT ?
            ''', (limit,))

            results = []
            for row in cursor.fetchall():
                url, title, content, metadata_json, timestamp_str, quality_score, language, content_hash, source_domain = row
                results.append(ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    metadata=json.loads(metadata_json),
                    timestamp=datetime.fromisoformat(timestamp_str),
                    quality_score=quality_score,
                    language=language,
                    content_hash=content_hash,
                    source_domain=source_domain
                ))
            return results
        finally:
            conn.close()
            
    def mark_processed(self, content_hashes: List[str]):
        if not content_hashes:
            return
        conn = self._connect()
        try:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(content_hashes))
            cursor.execute(f'''
                UPDATE scraped_content
                SET processed = TRUE
                WHERE content_hash IN ({placeholders})
            ''', content_hashes)
            conn.commit()
        finally:
            conn.close()
    
    def get_domain_stats(self, domain: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pages_scraped, last_scraped, avg_quality, success_rate
                FROM domain_stats WHERE domain = ?
            ''', (domain,))
            result = cursor.fetchone()

            if not result:
                return None

            pages_scraped, last_scraped, avg_quality, success_rate = result
            return {
                "pages_scraped": pages_scraped,
                "last_scraped": datetime.fromisoformat(last_scraped) if last_scraped else None,
                "avg_quality": avg_quality,
                "success_rate": success_rate,
            }
        finally:
            conn.close()

class IntelligentScraper:
    """Main intelligent web scraping class."""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.quality_assessor = ContentQualityAssessor()
        self.database = WebContentDatabase(config.database_path)
        self.user_agent = UserAgent() if config.rotate_user_agents else None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        if config.cache_enabled:
            requests_cache.install_cache(
                'web_scraping_cache',
                expire_after=timedelta(hours=config.cache_duration_hours)
            )
        
        # Setup Selenium if needed
        self.selenium_driver = None
        if config.use_selenium:
            self._setup_selenium()
        
        # Session pool
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=config.concurrent_sessions)
        )
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver."""
        chrome_options = Options()
        if self.config.headless_mode:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        
        try:
            self.selenium_driver = webdriver.Chrome(options=chrome_options)
            self.logger.info("Selenium WebDriver initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
            self.config.use_selenium = False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with rotation if enabled."""
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        if self.user_agent:
            headers['User-Agent'] = self.user_agent.random
        else:
            headers['User-Agent'] = 'Mozilla/5.0 (Living-LLM Web Learner)'
        
        return headers
    
    def _should_skip_domain(self, domain: str) -> bool:
        """Check if domain should be skipped based on limits."""
        stats = self.database.get_domain_stats(domain)
        if not stats:
            return False
        
        # Check page limit
        if stats['pages_scraped'] >= self.config.max_pages_per_domain:
            return True
        
        # Check cooldown
        if stats['last_scraped']:
            time_since_last = datetime.now() - stats['last_scraped']
            if time_since_last < timedelta(hours=self.config.domain_cooldown_hours):
                return True
        
        return False
    
    @sleep_and_retry
    @limits(calls=60, period=60)  # Rate limiting decorator
    async def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape content from a single URL."""
        try:
            # Parse URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check domain limits
            if self._should_skip_domain(domain):
                self.logger.info(f"Skipping {domain} due to limits")
                return None
            
            # Random delay
            if self.config.use_random_delays:
                delay = random.uniform(self.config.min_delay, self.config.max_delay)
                await asyncio.sleep(delay)
            
            # Scrape content
            content = None
            if self.config.use_selenium and self.selenium_driver:
                content = self._scrape_with_selenium(url)
            else:
                content = await self._scrape_with_requests(url)
            
            if content:
                # Store in database
                self.database.store_content(content)
                self.logger.info(f"Scraped: {url} (quality: {content.quality_score:.2f})")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None
    
    async def _scrape_with_requests(self, url: str) -> Optional[ScrapedContent]:
        """Scrape using aiohttp requests."""
        try:
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in self.config.allowed_content_types):
                    return None
                
                html = await response.text()
                return self._process_html(url, html)
                
        except Exception as e:
            self.logger.error(f"Request error for {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str) -> Optional[ScrapedContent]:
        """Scrape using Selenium for JavaScript-heavy sites."""
        try:
            self.selenium_driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.selenium_driver, self.config.selenium_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source
            html = self.selenium_driver.page_source
            return self._process_html(url, html)
            
        except Exception as e:
            self.logger.error(f"Selenium error for {url}: {e}")
            return None
    
    def _process_html(self, url: str, html: str) -> Optional[ScrapedContent]:
        """Process HTML content and extract meaningful text."""
        try:
            # Use newspaper3k for better content extraction
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            # Fallback to BeautifulSoup if newspaper fails
            if not article.text or len(article.text) < self.config.min_content_length:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                # Extract text
                text_content = soup.get_text()
                title = soup.find('title')
                title = title.get_text() if title else ""
            else:
                text_content = article.text
                title = article.title or ""
            
            # Clean and validate content
            text_content = self._clean_text(text_content)
            
            if len(text_content) < self.config.min_content_length:
                return None
            
            if len(text_content) > self.config.max_content_length:
                text_content = text_content[:self.config.max_content_length]
            
            # Assess quality
            quality_score = self.quality_assessor.assess_quality(text_content, title)
            
            if quality_score < self.config.content_quality_threshold:
                return None
            
            # Create content object
            parsed_url = urlparse(url)
            content_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            scraped_content = ScrapedContent(
                url=url,
                title=title.strip(),
                content=text_content,
                metadata={
                    'word_count': len(text_content.split()),
                    'scraping_method': 'selenium' if self.config.use_selenium else 'requests',
                    'content_length': len(text_content)
                },
                timestamp=datetime.now(),
                quality_score=quality_score,
                language='en',  # TODO: Add language detection
                content_hash=content_hash,
                source_domain=parsed_url.netloc
            )
            
            return scraped_content
            
        except Exception as e:
            self.logger.error(f"Error processing HTML from {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        text = text.replace("\r", "\n")

        # remove very short lines first
        lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 10]
        text = "\n".join(lines)

        # now normalize whitespace but keep newlines
        text = re.sub(r'[^\w\s.,!?;:()\-\[\]"\']+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # collapse leftover weird spacing
        return text.strip()
    
    async def scrape_news_feeds(self) -> List[ScrapedContent]:
        """Scrape content from news feeds."""
        results = []
        
        for feed_url in self.config.news_feeds:
            try:
                self.logger.info(f"Processing news feed: {feed_url}")
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                # Extract URLs
                urls = []
                for entry in feed.entries[:20]:  # Limit per feed
                    if hasattr(entry, 'link'):
                        urls.append(entry.link)
                
                # Scrape articles
                tasks = [self.scrape_url(url) for url in urls]
                scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
                
                for content in scraped_contents:
                    if isinstance(content, ScrapedContent):
                        results.append(content)
                        
            except Exception as e:
                self.logger.error(f"Error processing feed {feed_url}: {e}")
        
        return results
    
    async def continuous_learning_cycle(self) -> AsyncGenerator[List[ScrapedContent], None]:
        """Continuous learning cycle that yields batches of new content."""
        while True:
            try:
                # Scrape news feeds
                news_content = await self.scrape_news_feeds()
                
                if news_content:
                    self.logger.info(f"Scraped {len(news_content)} articles from news feeds")
                    yield news_content
                
                # Wait before next cycle
                await asyncio.sleep(3600)  # 1 hour cycle
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning cycle: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    def get_training_batch(self, batch_size: int = 32) -> List[ScrapedContent]:
        """Get a batch of high-quality content for training."""
        return self.database.get_unprocessed_content(batch_size)
    
    def mark_content_processed(self, content_hashes: List[str]):
        """Mark content as processed after training."""
        self.database.mark_processed(content_hashes)
    
    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
        
        if self.selenium_driver:
            self.selenium_driver.quit()


# Example usage and testing
async def main():
    """Example usage of the intelligent scraper."""
    config = ScrapingConfig()
    scraper = IntelligentScraper(config)
    
    try:
        # Test single URL scraping
        test_url = "https://www.reuters.com/technology/"
        content = await scraper.scrape_url(test_url)
        if content:
            print(f"Scraped: {content.title}")
            print(f"Quality: {content.quality_score:.2f}")
            print(f"Length: {len(content.content)} chars")
        
        # Test news feed scraping
        news_content = await scraper.scrape_news_feeds()
        print(f"Scraped {len(news_content)} articles from news feeds")
        
        # Get training batch
        training_batch = scraper.get_training_batch(10)
        print(f"Training batch: {len(training_batch)} items")
        
    finally:
        await scraper.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())