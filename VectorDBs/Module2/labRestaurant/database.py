import logging
import chromadb
from chromadb.utils import embedding_functions
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=Config.CHROMA_HOST,
            port=Config.CHROMA_PORT
        )
        try:
            hb = self.client.heartbeat()
            logger.info(f"Connected to ChromaDB HTTP Server successfully. Heartbeat: {hb}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB server: {e}")
            raise

    def get_embedding_function(self):
        """Get the embedding function used for document vector representation."""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )

    def get_or_create_collection(self, name: str = Config.COLLECTION_NAME):
        """Fetch existing collection or initialize a new one (optionally resets first)."""
        if Config.RESET_DATABASE:
            try:
                self.client.delete_collection(name=name)
                logger.info(f"RESET_DATABASE is active: Dropped existing collection '{name}'.")
            except Exception:
                # Silently catch error if collection does not exist
                pass

        return self.client.get_or_create_collection(
            name=name,
            metadata={
                "description": "A collection for storing restaurant dishes and metadata",
                "hnsw:space": "cosine"
            },
            embedding_function=self.get_embedding_function()
        )
