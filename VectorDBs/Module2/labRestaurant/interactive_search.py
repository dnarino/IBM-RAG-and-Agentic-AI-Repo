import json
import logging
from .database import DatabaseManager
from .parser import load_food_data
from .search import (
    populate_similarity_collection,
    perform_similarity_search,
    perform_filtered_similarity_search
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize database connection and fetch/create collection
        db_manager = DatabaseManager()
        collection = db_manager.get_or_create_collection()

        # Check existing collection document count
        count = collection.count()
        logger.info(f"Current collection contains {count} documents.")
        
        # If database is empty, load the dataset and populate it
        if count == 0:
            logger.info("Collection is empty. Starting ingestion process...")
            food_items = load_food_data()
            if food_items:
                populate_similarity_collection(collection, food_items)
                logger.info(f"Ingestion complete. Total items in database: {collection.count()}")
            else:
                logger.error("No food data found. Aborting startup.")
                return

        print("\n" + "="*60)
        print("🍳 WELCOME TO THE RESTAURANT VECTOR SEARCH INTERACTIVE CLI 🍳")
        print("="*60)
        print("Type 'exit' or 'quit' to terminate the session.\n")

        while True:
            query = input("🔍 Enter search query (e.g., 'sweet chocolate dessert'): ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting search session. Bon appétit!")
                break

            cuisine = input("🍝 Filter by Cuisine (press Enter to skip): ").strip() or None
            calories_raw = input("🔥 Max Calories limit (press Enter to skip): ").strip()
            
            max_calories = None
            if calories_raw:
                try:
                    max_calories = int(calories_raw)
                except ValueError:
                    print("⚠️ Invalid format for calories. Skipping calorie filtering.")

            print(f"\nSearching vector store for: '{query}'...")
            if cuisine or max_calories is not None:
                results = perform_filtered_similarity_search(
                    collection=collection,
                    query=query,
                    cuisine_filter=cuisine,
                    max_calories=max_calories
                )
            else:
                results = perform_similarity_search(
                    collection=collection,
                    query=query
                )

            if not results:
                print("❌ No matching records found.\n")
            else:
                print(f"✨ Top {len(results)} matching options:")
                print(json.dumps(results, indent=4))
                print("\n" + "-"*60 + "\n")

    except Exception as e:
        logger.error(f"Critical execution error: {e}")

if __name__ == "__main__":
    main()
