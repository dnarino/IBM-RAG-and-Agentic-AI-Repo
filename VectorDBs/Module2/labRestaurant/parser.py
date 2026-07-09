import json
import logging
from typing import List
from models import FoodItem
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_food_data(file_path: str = Config.DATASET_PATH) -> List[FoodItem]:
    """Load food data from JSON file and validate using FoodItem Pydantic schema."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            logger.error("Invalid dataset format: Root JSON structure must be a list.")
            return []

        food_items = []
        for i, raw_item in enumerate(raw_data):
            # Normalize food_id to string, fall back to index if missing
            if 'food_id' not in raw_item or raw_item['food_id'] is None:
                raw_item['food_id'] = str(i + 1)
            else:
                raw_item['food_id'] = str(raw_item['food_id'])

            # Validate and construct model
            food_item = FoodItem(**raw_item)
            food_items.append(food_item)

        logger.info(f"Successfully loaded and validated {len(food_items)} food items from {file_path}")
        return food_items

    except FileNotFoundError:
        logger.error(f"Dataset file not found at path: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON content: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error occurred during ingestion: {e}")
        return []
