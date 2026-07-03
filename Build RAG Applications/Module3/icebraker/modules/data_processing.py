import os
import logging
import chromadb
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import config

logger = logging.getLogger(__name__)

def split_profile_text(profile_text: str) -> List:
    """Splits the scraped raw profile text into Document nodes."""
    document = Document(text=profile_text)
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=50)
    return splitter.get_nodes_from_documents([document])

def create_vector_db(nodes: List) -> VectorStoreIndex:
    """Stores nodes in Chroma VectorStoreIndex."""
    # Globally configure the embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    chroma_client = chromadb.PersistentClient(path=config.db_persistence_path)
    
    # Reset collection to avoid cross-contamination of different profiles
    try:
        chroma_client.delete_collection(name="person_knowledge_base")
    except Exception:
        pass
        
    chroma_collection = chroma_client.get_or_create_collection(name="person_knowledge_base")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)
    return index

def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have valid embeddings inside Chroma."""
    try:
        vector_store = index.storage_context.vector_store
        
        # Check if we are using ChromaVectorStore (which wraps a native Chroma collection)
        if hasattr(vector_store, "_collection"):
            collection = vector_store._collection
            # Fetch all items in this collection
            result = collection.get(include=["embeddings"])
            node_ids = result.get("ids", [])
            embeddings_raw = result.get("embeddings")
            embeddings = embeddings_raw if embeddings_raw is not None else []
            
            if not node_ids or len(embeddings) == 0:
                logger.warning("The Chroma collection is empty. No nodes found to verify.")
                return False
                
            embeddings_map = dict(zip(node_ids, embeddings))
            missing_embeddings = False
            for node_id in node_ids:
                embedding = embeddings_map.get(node_id)
                if embedding is None:
                    logger.warning(f"Node ID {node_id} is missing its embedding in Chroma.")
                    missing_embeddings = True
                else:
                    logger.debug(f"Node ID {node_id} has a valid embedding.")
            
            if missing_embeddings:
                return False
            logger.info(f"Verification successful: {len(node_ids)} embeddings present and valid.")
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error in verify_embeddings: {e}")
        return False
