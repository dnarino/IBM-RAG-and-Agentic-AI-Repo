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


from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

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

#check apikey
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

#Add the embedding model

def openai_embedding() ->OpenAIEmbedding:
    try:
        check_api_key()
        embeddings= OpenAIEmbedding(
            model='text-embedding-3-small'
        )

        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Embeddings: {e}")
        raise
"""
Background

Before diving into the advanced retrieval techniques, let's understand the foundational concepts that make these retrievers powerful.
What are Advanced Retrievers?

Advanced retrievers in LlamaIndex are sophisticated components that go beyond simple vector similarity search to provide more nuanced, context-aware, and intelligent information retrieval. They combine multiple techniques such as:

    Semantic Understanding: Using embeddings to understand meaning and context
    Keyword Matching: Precise term-based search for exact specifications
    Hierarchical Context: Maintaining relationships between different levels of information
    Multi-Query Processing: Generating and combining results from multiple query variations
    Fusion Techniques: Intelligently combining results from different retrieval methods

Why are Advanced Retrievers Important?

    Improved Accuracy: Advanced retrievers can find more relevant information by using multiple search strategies
    Better Context Preservation: They maintain important relationships between pieces of information
    Reduced Hallucination: More precise retrieval leads to more accurate AI responses
    Scalability: Efficient retrieval strategies work better with large document collections
    Flexibility: Different retrieval methods can be combined for optimal results

Index Types Overview

Before exploring advanced retrievers, it's helpful to first understand the three main index types supported by LlamaIndex. Each is designed to support different retrieval scenarios:

VectorStoreIndex:

    Stores vector embeddings for each document chunk
    Best suited for semantic retrieval based on meaning
    Commonly used in LLM pipelines and RAG applications

DocumentSummaryIndex:

    Generates and stores summaries of documents at indexing time
    Uses summaries to filter documents before retrieving full content
    Especially useful for large and diverse document sets that cannot fit in the context window of an LLM

KeywordTableIndex:

    Extracts keywords from documents and maps them to specific content chunks
    Enables exact keyword matching for rule-based or hybrid search scenarios
    Ideal for applications requiring precise term matching

Sample Data Setup¶

We'll use a collection of AI and machine learning documents to demonstrate different retrieval strategies.
"""

def load_sample_data() -> tuple[list[str], dict[str, str]]:
    # Sample data for the lab - AI/ML focused documents
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
        "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
        "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
        "Unsupervised learning finds hidden patterns in data without labeled examples.",
        "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
        "Generative AI can create new content including text, images, code, and more.",
        "Large language models are trained on vast amounts of text data to understand and generate human-like text."
    ]

    # Consistent query examples used throughout the lab
    demo_queries = {
        "basic": "What is machine learning?",
        "technical": "neural networks deep learning", 
        "learning_types": "different types of learning",
        "advanced": "How do neural networks work in deep learning?",
        "applications": "What are the applications of AI?",
        "comprehensive": "What are the main approaches to machine learning?",
        "specific": "supervised learning techniques"
    }

    print(f"📄 Loaded {len(sample_documents)} sample documents")
    print(f"🔍 Prepared {len(demo_queries)} consistent demo queries")
    for i, doc in enumerate(sample_documents[:3], 1):
        print(f"{i}. {doc}")
    print("...")
    return sample_documents, demo_queries

class AdvanceRetrieversLab:
    def __init__(self):
        sample_documents, self.demo_queries = load_sample_data()
        self.documents = [Document(text=text) for text in sample_documents]
        rprint(self.documents[:1])


if __name__ == "__main__":
    lab = AdvanceRetrieversLab()