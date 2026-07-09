import logging
from config import Config
from database import DatabaseManager
from parser import load_food_data
from search import (
    populate_similarity_collection,
    perform_similarity_search,
    perform_filtered_similarity_search
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test():
    try:
        logger.info("Starting verification test...")
        db_manager = DatabaseManager()
        collection = db_manager.get_or_create_collection()
        
        # 1. Load data
        food_items = load_food_data()
        assert len(food_items) > 0, "Assertion failed: No food items loaded from JSON."
        
        # 2. Populate DB
        populate_similarity_collection(collection, food_items)
        assert collection.count() > 0, "Assertion failed: Collection count is 0 after population."
        logger.info("✅ Database population test passed.")
        
        # 3. Test basic semantic search
        results = perform_similarity_search(collection, "spicy soup", n_results=2)
        assert len(results) == 2, f"Assertion failed: Expected 2 search results, got {len(results)}."
        logger.info("✅ Semantic search test passed.")
        
        # 4. Test hybrid filtered search
        filtered_results = perform_filtered_similarity_search(
            collection,
            query="dessert",
            cuisine_filter="American",
            max_calories=400,
            n_results=1
        )
        assert len(filtered_results) <= 1, "Assertion failed: Filtered query returned more than 1 result."
        if filtered_results:
            assert filtered_results[0]['cuisine_type'] == "American", "Assertion failed: Cuisine filter failed to match 'American'."
            assert filtered_results[0]['food_calories_per_serving'] <= 400, "Assertion failed: Calorie limit filter failed."
        logger.info("✅ Hybrid filtered search test passed.")
        
        logger.info("🎉 All checks passed successfully! Modular code is clean and fully functional.")
    except Exception as e:
        logger.error(f"❌ Verification test failed: {e}")
        raise

if __name__ == "__main__":
    test()
