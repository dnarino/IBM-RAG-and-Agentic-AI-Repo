import re
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List

def extract_and_filter_url(results: str) -> List[str]:
    # Extract URLs from the DuckDuckGoSearchResults formatted string
    raw_urls = re.findall(r'link:\s*(https?://.*?)(?:,\s*(?:snippet|title):|$)', results)
    
    # Filter out domains that require login or are highly resistant to scraping
    blocked_domains = [
        "linkedin.com",
        "twitter.com",
        "t.co",
        "facebook.com",
        "instagram.com"
    ]
    
    filtered_urls = []
    for url in raw_urls:
        if not any(domain in url.lower() for domain in blocked_domains):
            filtered_urls.append(url)
            
    return filtered_urls


def scrape_person_profile(name: str) -> str:
    import logging
    logger = logging.getLogger(__name__)
    
    search = DuckDuckGoSearchResults()
    query = f"{name} professional profile biography career"
    try:
        results = search.run(query)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return ""
        
    urls = extract_and_filter_url(results)
    
    scraped_content = []
    for url in urls[:3]:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                scraped_content.append(soup.get_text().strip())
        except Exception as e:
            logger.debug(f"Failed to scrape {url} directly: {e}")
            
    # Fallback: if we couldn't scrape any external websites, use the rich search results themselves
    if not scraped_content and results:
        logger.info("Using search result snippets as fallback source.")
        scraped_content.append(results)
            
    return "\n\n".join(scraped_content)
