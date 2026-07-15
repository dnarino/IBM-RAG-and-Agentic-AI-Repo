import os
import sys
import logging
from typing import List
from dotenv import load_dotenv
from rich.logging import RichHandler
from rich import print as rprint

# Monkey-patch missing vector store classes in newer langchain-community packages dynamically
import langchain_community.vectorstores
import langchain_classic.vectorstores

for name in langchain_classic.vectorstores.__all__:
    if not hasattr(langchain_community.vectorstores, name):
        setattr(
            langchain_community.vectorstores,
            name,
            type(name, (object,), {})
        )

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

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

def self_query(vector_db: Chroma, query: str, llm: ChatOpenAI) -> List[Document]:
    try:
        logger.info(f"Setting up Self-Querying Retriever for query: '{query}'")
        
        metadata_field_info = [
            AttributeInfo(
                name="genre",
                description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
                type="string",
            ),
            AttributeInfo(
                name="year",
                description="The year the movie was released",
                type="integer",
            ),
            AttributeInfo(
                name="director",
                description="The name of the movie director",
                type="string",
            ),
            AttributeInfo(
                name="rating", 
                description="A 1-10 rating for the movie", 
                type="float"
            ),
        ]
        
        document_content_description = "Brief summary of a movie."
        
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_db,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info
        )
        
        logger.info("Executing self-query retrieval...")
        response = retriever.invoke(query)
        rprint("[bold green]✨ Search Results (Self-Query):[/bold green]")
        rprint(response)
        return response
    except Exception as e:
        logger.error(f"Error during self-query search: {e}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Initializing self-query search pipeline...")
        
        query = input("\n🔍 Enter search query (press Enter for default 'I want to watch a movie rated higher than 8.5'): ").strip()
        if not query:
            query = "What's a highly rated (above 8.5) science fiction film?"
            
        docs = [
            Document(
                page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
                metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
            ),
            Document(
                page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
                metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
            ),
            Document(
                page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
                metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
            ),
            Document(
                page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
                metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
            ),
            Document(
                page_content="Toys come alive and have a blast doing so",
                metadata={"year": 1995, "genre": "animated"},
            ),
            Document(
                page_content="Three men walk into the Zone, three men walk out of the Zone",
                metadata={
                    "year": 1979,
                    "director": "Andrei Tarkovsky",
                    "genre": "thriller",
                    "rating": 9.9,
                },
            ),
        ]
        
        logger.info("Indexing movie documents in Chroma...")
        movies_db = Chroma.from_documents(
            documents=docs,
            embedding=openai_embedding(),
            collection_name="movies_rating"
        )
        self_query(movies_db, query, llm=openAI_LLM())
        logger.info("Self-query pipeline completed successfully.")
                
    except Exception as e:
        logger.critical(f"Search pipeline failed: {e}")
        sys.exit(1)