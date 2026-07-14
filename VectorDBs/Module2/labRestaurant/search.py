import logging
from typing import List, Dict, Any, Optional
from .models import FoodItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_similarity_collection(collection, food_items: List[FoodItem]) -> None:
    """Populates ChromaDB collection with embeddings and metadata from a list of FoodItem models."""
    if not food_items:
        logger.warning("No food items provided to populate collection.")
        return

    documents = []
    metadatas = []
    ids = []
    seen_ids = set()

    for food in food_items:
        unique_id = food.get_deterministic_id()
        if unique_id in seen_ids:
            logger.warning(f"Found duplicate ID '{unique_id}' for food '{food.food_name}' in ingestion batch. Skipping to prevent DuplicateIDError.")
            continue
            
        seen_ids.add(unique_id)
        documents.append(food.to_embedding_text())
        ids.append(unique_id)
        metadatas.append({
            "name": food.food_name,
            "cuisine_type": food.cuisine_type,
            "ingredients": ", ".join(food.food_ingredients),
            "calories": food.food_calories_per_serving,
            "description": food.food_description,
            "cooking_method": food.cooking_method,
            "health_benefits": food.food_health_benefits,
            "taste_profile": food.taste_profile
        })

    try:
        # Using upsert to cleanly overwrite duplicates instead of throwing add errors
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully upserted {len(ids)} unique items to vector collection.")
    except Exception as e:
        logger.error(f"Error populating vector store collection: {e}")
        raise


def _format_search_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to convert raw ChromaDB query results into a structured list of dicts."""
    if not results or not results['ids'] or len(results['ids'][0]) == 0:
        return []

    formatted = []
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        similarity_score = 1.0 - distance
        
        metadata = results['metadatas'][0][i]
        formatted.append({
            'food_id': results['ids'][0][i],
            'food_name': metadata.get('name', ''),
            'food_description': metadata.get('description', ''),
            'cuisine_type': metadata.get('cuisine_type', 'Unknown'),
            'food_calories_per_serving': metadata.get('calories', 0),
            'similarity_score': similarity_score,
            'distance': distance
        })
    return formatted

def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Execute simple semantic query vector search."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return _format_search_results(results)
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        return []

def perform_filtered_similarity_search(collection, query: str, cuisine_filter: Optional[str] = None, 
                                     max_calories: Optional[int] = None, n_results: int = 5) -> List[Dict[str, Any]]:
    """Execute hybrid filtered similarity search with metadata constraints."""
    where_clause = None
    filters = []

    if cuisine_filter:
        filters.append({"cuisine_type": cuisine_filter})
    if max_calories:
        filters.append({"calories": {"$lte": max_calories}})

    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        return _format_search_results(results)
    except Exception as e:
        logger.error(f"Error performing filtered similarity search: {e}")
        return []
