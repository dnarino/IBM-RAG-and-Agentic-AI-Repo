import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host="localhost", port=8000)

# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

# 2. Define the embedding model 
# IMPORTANT: Since you are using a thin HTTP client, it's best practice 
# to explicitly specify your embedding function.
embedding_function = ONNXMiniLM_L6_V2()

collection=client.get_or_create_collection(
     name="docker_knowledge_base",
     embedding_function=embedding_function
)

# 4. Prepare your document data chunks
documents = [
    "Docker containers encapsulate software and its dependencies into isolated units.",
    "FastAPI leverages Pydantic for data validation and auto-generates OpenAPI documentation.",
    "Chroma DB utilizes HNSW indexing under the hood for millisecond proximity operations.",
    "Bogotá is the high-altitude, bustling capital city of Colombia."
]
   
ids = ["id_docker", "id_fastapi", "id_chroma", "id_colombia"]
metadatas = [
    {"topic": "devops"},
    {"topic": "backend"},
    {"topic": "databases"},
    {"topic": "geography"}
]

collection.upsert(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print("✅ Documents successfully sent and indexed inside the Docker container!\n")

# 6. Execute semantic query search over the network
query_text = "How can I build clean APIs or handle backend validation?"
print(f"🔍 Querying Docker container for: '{query_text}'")

results = collection.query(
    query_texts=[query_text],
    n_results=1
)

# 7. Output top match
print("-" * 60)
print(f"Matched ID: {results['ids'][0][0]}")
print(f"Content:    {results['documents'][0][0]}")
print(f"Distance:   {results['distances'][0][0]:.4f}")
print("-" * 60)