import os

class Config:
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    RESET_DATABASE: bool = os.getenv("RESET_DATABASE", "False").lower() in ("true", "1", "t", "yes", "y", "1")
    
    # Dataset path resolved dynamically relative to this file
    DATASET_PATH: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "FoodDataSet.json")
    )
    
    COLLECTION_NAME: str = "restaurant_collection"
