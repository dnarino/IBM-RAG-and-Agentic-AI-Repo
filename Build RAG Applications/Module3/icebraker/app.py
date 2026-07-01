import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from config import (
    template_facts,
    template_user_questions,
    PARAMETERS,
    OPENAI_MODEL_ID,
    CHUNK_SIZE,
    SIMILARITY_TOP_K,
    db_persistence_path
)
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# LlamaIndex native OpenAI LLM client
from llama_index.llms.openai import OpenAI

# pyrefly: ignore [missing-import]
from modules.helper import search_and_scrape_person, verify_embeddings

load_dotenv()

def main():
    # ----------------------------------------------------
    # 1: Changing the Default LLM
    # ----------------------------------------------------
    print("\n--- Topic 1: Changing the Default LLM ---")
    # Initialize the native LlamaIndex OpenAI client using the loaded env variables
    openai_llm = OpenAI(
        model=OPENAI_MODEL_ID,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=PARAMETERS.get("tempeture", 0.2),
        max_tokens=PARAMETERS.get("max_new_tokens", 256)
    )

    
    # ----------------------------------------------------
    # 2. Base Setup (DB Load & Model Config)
    # ----------------------------------------------------
    print("Scraping person info...")
    scraping_results = search_and_scrape_person('donald trump')
    
    # Create data directory and save results to a text file
    os.makedirs("data", exist_ok=True)
    with open("data/scraped_profile.txt", "w", encoding="utf-8") as f:
        f.write(scraping_results)
    
    # ----------------------------------------------------
    # 3. Base Setup (DB Load & Model Config)
    # ----------------------------------------------------
    print("Loading embedding model...")
    # Globally configure the embedding model (to ensure both index & query use it)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = openai_llm
    
    # Define the chunking and splitter settings globally
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
    
    chroma_client = chromadb.PersistentClient(path=db_persistence_path)
    chroma_collection = chroma_client.get_or_create_collection(name="person_knowledge_base")

    # Load index from the existing vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load the documents from the saved text file
    documents = SimpleDirectoryReader(input_files=["data/scraped_profile.txt"]).load_data()
    
    # Build the index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    print("Index successfully built!")
    
    # Verify embeddings logic
    verify_embeddings(index)
    
    # Quick Verification Query
    query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
    response = query_engine.query("What are some professional details about Donald Trump?")
    print(f"\nTest Query Response:\n{response}")


if __name__ == "__main__":
    main()

