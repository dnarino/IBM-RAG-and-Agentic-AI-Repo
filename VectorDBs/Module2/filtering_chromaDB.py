import ssl
# Bypass corporate SSL certificate verification issues on Hugging Face downloads
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
from chromadb.utils import embedding_functions


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host="localhost", port=8000)

# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

# Create or get Collection
collection = client.get_or_create_collection(
    name="filter_demo",
    metadata={"description": "Used to demo filtering in ChromaDB"},
    embedding_function=ef
)

print(f"Collection: {collection.name}")

collection.upsert(
    documents=[
        "This is a document about LangChain",
        "This is a reading about LlamaIndex",
        "This is a book about Python",
        "This is a document about pandas",
        "This is another document about LangChain"
    ],
    metadatas=[
        {"source": "langchain.com", "version": 0.1},
        {"source": "llamaindex.ai", "version": 0.2},
        {"source": "python.org", "version": 0.3},
        {"source": "pandas.pydata.org", "version": 0.4},
        {"source": "langchain.com", "version": 0.5},
    ],
    ids=["id1", "id2", "id3", "id4", "id5"]
)

# Filter using Metadata
filtered_results = collection.get(
    where={"source": {"$eq": "langchain.com"}}
)

print("\n🔍 FILTERED RESULTS (Where source = langchain.com):")
print(filtered_results["documents"])

# we were only interested in LangChain documents with versions less than 0.3.

filtered_and_results = collection.get(
    where={
        "$and":[
            {"source":{"$eq":"langchain.com"}},
            {"version":{"$lt":0.3}}
        ]
    }
)

print("\n🔍 FILTERED RESULTS (Where source = langchain.com AND version < 0.3):")
print(filtered_and_results["documents"])

#The following retrieves all documents about LangChain and LlamaIndex with a version less than 0.3:

filtered_in_results = collection.get(
    where={
        "$and":[
            {"source":{"$in":["langchain.com", "llamaindex.ai"]}},
            {"version":{"$lt":0.3}}
        ]
    }
)

print("\n🔍 FILTERED RESULTS (Where source IN [langchain.com, llamaindex.ai] AND version < 0.3):")
print(filtered_in_results["documents"])