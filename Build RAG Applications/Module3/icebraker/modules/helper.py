import re
import logging
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from typing import List
from llama_index.core import VectorStoreIndex

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded in the vector store.
    
    Args:
        index: VectorStoreIndex to verify.
        
    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        vector_store = index.storage_context.vector_store
        missing_embeddings = False
        
        # Check if we are using ChromaVectorStore (which wraps a native Chroma collection)
        if hasattr(vector_store, "_collection"):
            collection = vector_store._collection
            # Fetch all items in this collection
            result = collection.get(include=["embeddings"])
            node_ids = result.get("ids", [])
            embeddings_raw = result.get("embeddings")
            embeddings = embeddings_raw if embeddings_raw is not None else []
            
            if not node_ids:
                logger.warning("The Chroma collection is empty. No nodes found to verify.")
                return False
                
            embeddings_map = dict(zip(node_ids, embeddings))
            
            for node_id in node_ids:
                embedding = embeddings_map.get(node_id)
                if embedding is None:
                    logger.warning(f"Node ID {node_id} is missing its embedding in Chroma.")
                    missing_embeddings = True
                else:
                    logger.debug(f"Node ID {node_id} has a valid embedding.")
                    
        # Fallback for LlamaIndex's default in-memory SimpleVectorStore
        elif hasattr(vector_store, "data") and hasattr(vector_store.data, "embedding_dict"):
            embedding_dict = vector_store.data.embedding_dict
            node_ids = list(embedding_dict.keys())
            
            if not node_ids:
                logger.warning("The SimpleVectorStore is empty. No nodes found to verify.")
                return False
                
            for node_id in node_ids:
                embedding = embedding_dict.get(node_id)
                if embedding is None:
                    logger.warning(f"Node ID {node_id} is missing its embedding in SimpleVectorStore.")
                    missing_embeddings = True
                else:
                    logger.debug(f"Node ID {node_id} has a valid embedding of size {len(embedding)}.")
                    
        else:
            # General fallback using index struct if available
            node_ids = list(index.index_struct.nodes_dict.keys())
            if not node_ids:
                logger.warning("The index structure and vector store mapping are empty.")
                return False
                
            logger.info("Non-standard vector store detected. Attempting fallback node check...")
            for node_id in node_ids:
                try:
                    nodes = vector_store.get_nodes(node_ids=[node_id])
                    if not nodes or nodes[0].embedding is None:
                        logger.warning(f"Node ID {node_id} has a None embedding.")
                        missing_embeddings = True
                except Exception as node_err:
                    logger.warning(f"Could not verify node ID {node_id}: {node_err}")
                    missing_embeddings = True

        if missing_embeddings:
            logger.warning("Verification complete: Some node embeddings are missing or invalid.")
            return False
        
        logger.info(f"Verification complete: All {len(node_ids)} node embeddings are valid and present.")
        return True

    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}", exc_info=True)
        return False


