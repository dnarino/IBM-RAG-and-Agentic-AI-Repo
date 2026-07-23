from typing import Tuple
import logging
import os
import ssl
import sys

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Bypass SSL verification issues for dataset/model downloads
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context
ssl.create_default_context = ssl._create_unverified_context

from pprint import pprint
import re
import faiss
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import tensorflow_hub as hub

# 1. Fetch Dataset
try:
    logger.info("Fetching 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all')
    documents = newsgroups.data
    logger.info(f"Successfully loaded {len(documents)} raw documents.")
except Exception as e:
    logger.error(f"Failed to download or load 20 Newsgroups dataset: {e}")
    sys.exit(1)

# 2. Text Preprocessing Function
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

logger.info("Preprocessing document text...")
processed_documents = [preprocess_text(doc) for doc in documents]

# Display sample comparison
sample_index = 0
print("\n" + "=" * 60)
print("ORIGINAL POST SAMPLE:")
print("=" * 60)
print(documents[sample_index][:250] + "...\n")

print("=" * 60)
print("PREPROCESSED POST SAMPLE:")
print("=" * 60)
print(processed_documents[sample_index][:250] + "...\n")

# 3. Load Universal Sentence Encoder (USE)
try:
    logger.info("Loading Universal Sentence Encoder model from TF-Hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logger.info("USE model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load USE model from TF-Hub: {e}")
    sys.exit(1)

def embed_text(text_list: list[str]) -> np.ndarray:
    try:
        return embed(text_list).numpy()
    except Exception as e:
        logger.error(f"Error generating embeddings for text batch: {e}")
        raise e

# 4. Generate Embeddings with Chunked Batching (1,000 docs per batch)
batch_size = 1000
X_use_list = []
logger.info(f"Generating USE embeddings for {len(processed_documents)} documents in batches of {batch_size}...")

try:
    for i in range(0, len(processed_documents), batch_size):
        batch_docs = processed_documents[i : i + batch_size]
        batch_vecs = embed_text(batch_docs)
        X_use_list.append(batch_vecs)

    X_use = np.vstack(X_use_list)
    logger.info(f"Embedding matrix constructed successfully with shape: {X_use.shape}")
except Exception as e:
    logger.error(f"Failed during batch embedding generation: {e}")
    sys.exit(1)

# 5. Indexing with FAISS
try:
    dimension = X_use.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(X_use)

    logger.info(f"FAISS indexing complete. Total vectors indexed: {index.ntotal}")
    logger.info(f"Vector dimension size: {index.d}")
    logger.info(f"Is index trained? {index.is_trained}")
except Exception as e:
    logger.error(f"Failed to create or populate FAISS index: {e}")
    sys.exit(1)

# 6. Querying Function with Error Handling
def search(query_text: str, k: int = 5)-> Tuple[np.ndarray,np.ndarray]:
    try:
        preprocessed_query = preprocess_text(query_text)
        if not preprocessed_query:
            logger.warning(f"Query string '{query_text}' resulted in an empty string after preprocessing.")
            return np.array([]), np.array([])
        
        query_vector = embed_text([preprocessed_query])
        distances, indices = index.search(query_vector.astype('float32'), k)
        return distances, indices
    except Exception as e:
        logger.error(f"Error executing vector search for query '{query_text}': {e}")
        return np.array([]), np.array([])

# 7. Execute Query Example
query_text = 'motorcycle'
logger.info(f"Executing search query: '{query_text}'")
distances, indices = search(query_text, k=5)

if len(indices) > 0 and len(indices[0]) > 0:
    print("\n" + "=" * 60)
    print(f"SEARCH RESULTS FOR QUERY: '{query_text}'")
    print("=" * 60)

    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}: (Distance: {distances[0][i]:.4f})")
        print(f"{processed_documents[idx][:250]}...\n")
else:
    logger.warning("No search results were retrieved.")
