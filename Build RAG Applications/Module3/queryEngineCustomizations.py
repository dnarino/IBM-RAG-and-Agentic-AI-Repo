import os
import chromadb
# Import core module to load environment variables (.env) and suppress warnings
import core

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LlamaIndex native OpenAI LLM client
from llama_index.llms.openai import OpenAI

def main():
    # ----------------------------------------------------
    # 1. Base Setup (DB Load & Model Config)
    # ----------------------------------------------------
    print("Loading embedding model...")
    # Globally configure the embedding model (to ensure both index & query use it)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Connecting to ChromaDB database...")
    # Locate the existing chroma_db from Module1 (relative path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_persistence_path = os.path.join(base_dir, "../Module1/chroma_db")
    
    chroma_client = chromadb.PersistentClient(path=db_persistence_path)
    chroma_collection = chroma_client.get_or_create_collection(name="enterprise_knowledge_base")

    # Load index from the existing vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # ----------------------------------------------------
    # Topic 1: Changing the Default LLM
    # ----------------------------------------------------
    print("\n--- Topic 1: Changing the Default LLM ---")
    # Initialize the native LlamaIndex OpenAI client using the loaded env variables
    openai_llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
    # We can pass the custom LLM directly to the query engine
    query_engine_custom_llm = index.as_query_engine(llm=openai_llm)
    
    prompt = "¿Cuál es el monto del estipendio en la política móvil?"
    print(f"Query: {prompt}")
    response = query_engine_custom_llm.query(prompt)
    print(f"Response (Custom LLM): {response}\n")

    # ----------------------------------------------------
    # Topic 2: Defining a Custom Prompt Template
    # ----------------------------------------------------
    print("--- Topic 2: Defining a Custom Prompt Template ---")
    from llama_index.core import PromptTemplate

    # A custom prompt enforcing a specific tone and response pattern
    qa_template_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, please answer this question: {query_str}\n"
        "Make sure to formulate the response in a polite, highly formal corporate style."
    )
    qa_template = PromptTemplate(qa_template_str)
    
    # Pass the prompt template to the engine via text_qa_template
    query_engine_custom_prompt = index.as_query_engine(
        llm=openai_llm, 
        text_qa_template=qa_template
    )
    
    print(f"Query: {prompt}")
    response = query_engine_custom_prompt.query(prompt)
    print(f"Response (Custom Prompt): {response}\n")

    # ----------------------------------------------------
    # Topic 3: Specifying a Custom Retriever
    # ----------------------------------------------------
    print("--- Topic 3: Specifying a Custom Retriever ---")
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever

    # Explicitly construct a custom retriever (limiting search results to top 1 matched chunk)
    custom_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1  # Only fetch the single most relevant chunk
    )
    
    # Wrap the custom retriever in a RetrieverQueryEngine
    query_engine_custom_retriever = RetrieverQueryEngine(
        retriever=custom_retriever,
        node_postprocessors=[]  # We can optionally add re-rankers here!
    )
    
    # Set the LLM globally since RetrieverQueryEngine uses the global Settings.llm
    Settings.llm = openai_llm

    print(f"Query: {prompt}")
    response = query_engine_custom_retriever.query(prompt)
    print(f"Response (Custom Retriever): {response}\n")

if __name__ == "__main__":
    main()
