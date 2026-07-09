import ssl
import os
import json
import re
import numpy as np 
from typing import List, Dict
ssl._create_default_https_context=ssl._create_unverified_context

import chromadb
from chromadb.utils import embedding_functions

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
RESET_DATABASE = False  # Set to True only when you need to wipe/rebuild the index
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "FoodDataSet.json")


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

collection_name = "restaurant_collection"

#dataloading function
def load_food_data(file_path: str) -> List[Dict]:
    """Load food data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            food_data = json.load(file)
        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(food_data):
            # Normalize food_id to string
            if 'food_id' not in item:
                item['food_id'] = str(i + 1)
            else:
                item['food_id'] = str(item['food_id'])
            
            # Ensure required fields exist
            if 'food_ingredients' not in item:
                item['food_ingredients'] = []
            if 'food_description' not in item:
                item['food_description'] = ''
            if 'cuisine_type' not in item:
                item['cuisine_type'] = 'Unknown'
            if 'food_calories_per_serving' not in item:
                item['food_calories_per_serving'] = 0
            
            # Extract taste features from nested food_features if available
            if 'food_features' in item and isinstance(item['food_features'], dict):
                taste_features = []
                for key, value in item['food_features'].items():
                    if value:
                        taste_features.append(str(value))
                item['taste_profile'] = ', '.join(taste_features)
            else:
                item['taste_profile'] = ''
        
        print(f"Successfully loaded {len(food_data)} food items from {file_path}")
        return food_data
        
        
    except Exception as e:
        print(f"Error loading food data: {e}")
        return []


def main()-> None:
    try:
        food_data = load_food_data(DATASET_PATH)
        print(json.dumps(food_data[0], indent=4)) 
    except Exception as e:
        print(f"Error in main() :{e}")

if __name__== "__main__":
    main()