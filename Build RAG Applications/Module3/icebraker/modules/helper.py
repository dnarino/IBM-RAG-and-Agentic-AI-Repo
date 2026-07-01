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


def search_and_scrape_person(name:str):
    search= DuckDuckGoSearchResults()
    query= f"{name} professional profile"
    # 1. Fetch URLs
    results = search.run(query)
    # Parse results to extract URLs (filtering out linkedin.com, twitter.com)
    urls= extract_and_filter_url(results)

    scraped_content= []
    # 2. Scrape only the allowed URLs
    for url in urls[:2]:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if response.status_code==200:
                soup =BeautifulSoup(response.text,'html.parser')
                scraped_content.append(soup.get_text())
        except Exception as e:
            print(f"Failed to scrape{url}:{e}")
    return "\n\n".join(scraped_content)

