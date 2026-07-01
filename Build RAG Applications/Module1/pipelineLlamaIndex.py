import os
import chromadb

import core
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuración de LLM personalizada usando la función de core/openai_client
from core.openai_client import get_openai_chat
from llama_index.llms.langchain import LangChainLLM

Settings.llm = LangChainLLM(llm=get_openai_chat())

def run_integrated_rag_pipeline(input_dir_path: str, db_persistence_path: str) -> str:
    """Ejecuta un ciclo completo de RAG."""
    print("Ingestando archivos desde el sistema...")
    if not os.path.exists(input_dir_path):
        os.makedirs(input_dir_path)
        with open(os.path.join(input_dir_path, "company_policy.txt"), "w") as f:
            f.write("Company Mobile Policy: Employees are eligible for a $50/month device stipend if remote.")

    reader = SimpleDirectoryReader(input_dir=input_dir_path, recursive=True)
    raw_documents = reader.load_data()

    print("Segmentando documentos en nodos semánticos...")
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    processed_nodes = node_parser.get_nodes_from_documents(raw_documents)

    print("Instanciando modelos de embedding y almacenamiento...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path=db_persistence_path)
    chroma_collection = chroma_client.get_or_create_collection(name="enterprise_knowledge_base")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Construyendo VectorStoreIndex...")
    index = VectorStoreIndex(nodes=processed_nodes, storage_context=storage_context)

    print("Empaquetando pipeline en QueryEngine...")
    query_engine = index.as_query_engine(similarity_top_k=3)

    test_prompt = "¿Cuál es el monto del estipendio en la política móvil?"
    response = query_engine.query(test_prompt)
    return str(response)

if __name__ == "__main__":
    DATA_DIR = "./workspace_data"
    CHROMA_DIR = "./chroma_db"
    
    pipeline_result = run_integrated_rag_pipeline(DATA_DIR, CHROMA_DIR)
    print(f"\nRespuesta: {pipeline_result}")
