import os
import sys
import logging
from typing import List
from dotenv import load_dotenv
from rich.logging import RichHandler
from rich import print as rprint

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.stores import InMemoryStore


# Configure logging using RichHandler for beautiful output logs
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("smarter_search")

# Suppress debug logs from HTTP clients and other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

load_dotenv()

def check_api_key() -> None:
    """Ensure the OpenAI API key is present in environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-proj-YOUR"):
        raise ValueError(
            "Missing or placeholder OpenAI API Key! Please verify that OPENAI_API_KEY "
            "is correctly configured in your '.env' file at the root of the project."
        )

# Add the LLM
def openAI_LLM() -> ChatOpenAI:
    try:
        check_api_key()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=256,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}")
        raise

# Add the text splitter
def text_splitter(data: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(data)
        logger.info(f"Successfully split document into {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error during text splitting: {e}")
        raise

# Add the embedding model
def openai_embedding() -> OpenAIEmbeddings:
    try:
        check_api_key()
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small'
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Embeddings: {e}")
        raise

def load_data() -> List[Document]:
    try:
        file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "companyPolicies.txt")
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
            
        loader = TextLoader(file_path)
        txt_data = loader.load()
        logger.info("Successfully loaded companyPolicies.txt document.")
        return txt_data
    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading policies data: {e}")
        raise

def load_pdf() -> List[Document]:
    pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf"
    try:
        logger.info(f"Fetching PDF document from network URL...")
        loader = PyPDFLoader(pdf_url)
        pdf_data = loader.load()
        logger.info("Successfully loaded PDF document.")
        return pdf_data
    except Exception as e:
        logger.error(
            f"Failed to download or parse PDF from {pdf_url}. "
            f"Please verify your network connection. Error details: {e}"
        )
        raise

def run_similarity_search(vector_db: Chroma, query: str) -> List[Document]:
    try:
        logger.info(f"Running Standard Similarity Search for query: '{query}'")
        retriever = vector_db.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(query)
        rprint("[bold green]✨ Search Results (Standard Similarity):[/bold green]")
        rprint(docs)
        return docs
    except Exception as e:
        logger.error(f"Error during standard similarity search: {e}")
        raise

def run_mmr_search(vector_db: Chroma, query: str) -> List[Document]:
    try:
        logger.info(f"Running Maximal Marginal Relevance (MMR) Search for query: '{query}'")
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3, 
                "fetch_k": 20,
                "score_threshold": 0.20
            }
        )
        docs = retriever.invoke(query)
        rprint("[bold green]✨ Search Results (MMR):[/bold green]")
        rprint(docs)
        return docs
    except Exception as e:
        logger.error(f"Error during MMR search: {e}")
        raise

def run_similarity_threshold_search(vector_db: Chroma, query: str) -> List[Document]:
    try:
        logger.info(f"Running Similarity Score Threshold Search for query: '{query}'")
        retriever = vector_db.as_retriever(
            search_kwargs={"score_threshold": 0.4},
            search_type="similarity_score_threshold",
        )
        docs = retriever.invoke(query)
        rprint("[bold green]✨ Search Results (Similarity Threshold):[/bold green]")
        rprint(docs)
        return docs
    except Exception as e:
        logger.error(f"Error during similarity threshold search: {e}")
        raise

def run_multi_query_search(vector_db: Chroma, query: str, llm: ChatOpenAI) -> List[Document]:
    try:
        logger.info(f"Running Multi-Query Search for query: '{query}'")
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(), llm=llm
        )
        docs = retriever.invoke(query)
        rprint("[bold green]✨ Search Results (Multi-Query):[/bold green]")
        rprint(docs)
        return docs
    except Exception as e:
        logger.error(f"Error during multi-query search: {e}")
        raise
def run_parent_document_search(query: str, data: List[Document], collection_name: str) -> List[Document]:
    try:
        logger.info(f"Running Parent Document Search for query: '{query}' in collection '{collection_name}'")
        # Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
        parent_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
        child_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='\n')
        
        vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=openai_embedding(),
        )
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_db,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        logger.info("Adding documents to Parent Document Retriever...")
        retriever.add_documents(data)
        
        logger.info("Executing retrieval query...")
        docs = retriever.invoke(query)
        
        rprint(f"[bold green]✨ Search Results (Parent Document Retriever - {collection_name}):[/bold green]")
        rprint(docs)
        return docs
    except Exception as e:
        logger.error(f"Error during Parent Document search: {e}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Initializing search pipeline...")
        
        while True:
            rprint("\n[bold cyan]=========================================[/bold cyan]")
            rprint("[bold cyan]      Smarter Search Retrieval Lab       [/bold cyan]")
            rprint("[bold cyan]=========================================[/bold cyan]")
            rprint("1. Query Company Policies (txt)")
            rprint("2. Query LangChain Research Paper (pdf)")
            rprint("3. Exit")
            
            try:
                choice = input("\nSelect option (1-3): ").strip()
            except (KeyboardInterrupt, EOFError):
                rprint("\n[bold yellow]Exiting...[/bold yellow]")
                break
                
            if choice == "1":
                rprint("\n[bold blue]📖 Loading policies document...[/bold blue]")
                data = load_data()
                chunks = text_splitter(data, 200, 20)
                
                rprint("[bold blue]🧠 Generating embeddings & indexing in Chroma...[/bold blue]")
                policies_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=openai_embedding(),
                    collection_name="company_policies"
                )
                
                query = input("\n🔍 Enter search query (press Enter for default 'email policy'): ").strip()
                if not query:
                    query = "email policy"
                
                run_similarity_search(policies_db, query)
                run_mmr_search(policies_db, query)
                run_similarity_threshold_search(policies_db, query)
                run_parent_document_search(query, data, "split_parents_policies")
                
            elif choice == "2":
                rprint("\n[bold blue]📄 Loading LangChain research paper PDF...[/bold blue]")
                pdf_data = load_pdf()
                chunks_pdf = text_splitter(pdf_data, 500, 20)
                
                rprint("[bold blue]🧠 Generating embeddings & indexing in Chroma...[/bold blue]")
                pdf_db = Chroma.from_documents(
                    documents=chunks_pdf,
                    embedding=openai_embedding(),
                    collection_name="research_papers"
                )
                
                query = input("\n🔍 Enter search query (press Enter for default 'What does the paper say about langchain?'): ").strip()
                if not query:
                    query = "What does the paper say about langchain?"
                
                run_similarity_search(pdf_db, query)
                run_mmr_search(pdf_db, query)
                run_similarity_threshold_search(pdf_db, query)
                run_multi_query_search(pdf_db, query, openAI_LLM())
                run_parent_document_search(query, pdf_data, "split_parents_pdf")
                
            elif choice == "3":
                rprint("\n[bold yellow]Exiting... Goodbye![/bold yellow]")
                break
            else:
                rprint("[bold red]Invalid option! Please enter 1, 2, or 3.[/bold red]")
                
    except Exception as e:
        logger.critical(f"Search pipeline failed: {e}")
        sys.exit(1)