import os
import json
from typing import List, Optional
import asyncio
import numpy as np
import logging
from rich.logging import RichHandler
from rich import print as rprint
from dotenv import load_dotenv

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)

from llama_index.core.retrievers import(
    BaseRetriever,
    VectorIndexRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)

from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)

from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI

from sentence_transformers import SentenceTransformer

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
def openAI_LLM() -> OpenAI:
    try:
        check_api_key()
        llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=256,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}")
        raise