from certifi import where
import ssl
# Bypass corporate SSL certificate verification issues on Hugging Face downloads
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
from chromadb.utils import embedding_functions

# Initialize the embedding model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host="localhost", port=8000)

# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

collection= client.get_or_create_collection(
    name="my_collection_name",
    metadata={"topic":"query testing"},
    configuration={
        "hnsw":{
            "space":"cosine",
            "ef_search":100,
            "ef_construction":100,
            "max_neighbors":16
        },
        "embedding_function":ef
    }
)

#Add data

collection.upsert(
   documents=[
       "Giant pandas are a bear species that lives in mountainous areas.",
       "A pandas DataFrame stores two-dimensional, tabular data",
       "I think everyone agrees that pandas are some of the cutest animals on the planet",
       "A direct comparison between pandas and polars indicates that polars is a more efficient library than pandas.",
   ],
   metadatas=[
       {"topic": "animals"},
       {"topic": "data analysis"},
       {"topic": "animals"},
       {"topic": "data analysis"},
   ],
   ids=["id1", "id2", "id3", "id4"]
)

#Let's query our collection using the query cats:

filtered_results = collection.query(
    query_texts=["cats"],
    n_results=10
)

print("\n🔍 FILTERED RESULTS  query_texts=['cats']")
print(filtered_results["documents"])

#quering with filters

filtered_results=collection.query(
    query_texts=["polar bears"],
    n_results=1,
    where={'topic':'animals'}
)

print("\n🔍 FILTERED RESULTS  query_texts=['cats']")
print(filtered_results["documents"])

filtered_results=collection.query(
    query_texts=["polar bears"],
    n_results=1,
    where_document={'$not_contains':'library'}
)

print("\n🔍 FILTERED RESULTS  query_texts=['cats']")
print(filtered_results["documents"])

filtered_results=collection.query(
    query_texts=["polar bears"],
    n_results=1,
    where={'topic':'animals'},
    where_document={'$not_contains':'library'}
)

print("\n🔍 THE MOST FILTERED RESULTS  query_texts=['cats']")
print(filtered_results["documents"])

