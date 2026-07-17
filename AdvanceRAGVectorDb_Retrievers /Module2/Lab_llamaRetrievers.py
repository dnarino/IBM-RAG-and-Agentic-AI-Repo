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
        self.nodes= SentenceSplitter().get_nodes_from_documents(self.documents)
         # 1. Explicitly configure LlamaIndex to use your custom models
        logger.info("Configuring custom LLM and Embeddings...")
        Settings.llm = openAI_LLM()                 
        Settings.embed_model = openai_embedding()   
        print("📊 Creating indexes...")
        # Create various indexes
        self.vector_index= VectorStoreIndex.from_documents(self.documents)
        self.document_summary_index=DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index= KeywordTableIndex.from_documents(self.documents)

        print("✅ Advanced Retrievers Lab Initialized!")
        print(f"📄 Loaded {len(self.documents)} documents")
        print(f"🔢 Created {len(self.nodes)} nodes")

    def vector_index_retriever(self, query: str) -> List[NodeWithScore]:
        """
        1. Vector Index Retriever - The Foundation

The Vector Index Retriever uses vector embeddings to find semantically related content, making it ideal for general-purpose search and widely used in retrieval-augmented generation (RAG) pipelines.

How it works:

    Documents are split into nodes and embedded using the configured embedding model
    Query is converted to an embedding vector
    Returns nodes ranked by cosine similarity to the query embedding
    Generates embeddings in batches of 2048 nodes by default

When to use:

    General-purpose semantic search (most common use case)
    Finding conceptually related content based on meaning rather than exact keywords
    RAG pipelines where semantic understanding is crucial
    When exact keyword matching isn't the primary requirement

Key characteristics from authoritative source:

    Stores embeddings for each document chunk (VectorStoreIndex foundation)
    Best for semantic retrieval based on meaning and context
    Commonly used in LLM pipelines for retrieval-augmented generation

Strengths:

    Excellent semantic understanding and context awareness
    Handles synonyms and related concepts effectively
    Works well with natural language queries

Limitations:

    May miss exact keyword matches when specific terms are crucial
    Requires a good embedding model for optimal performance
    Can be computationally intensive for large document collections
"""
        try:
            rprint("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
            rprint("[bold cyan]            1. VECTOR INDEX RETRIEVER                      [/bold cyan]")
            rprint("[bold cyan]" + "=" * 60 + "[/bold cyan]")
            vector_retriever = VectorIndexRetriever(
                index=self.vector_index,
                similarity_top_k=3
            )
            return vector_retriever.retrieve(query)
        except Exception as e:
            logger.error(f"Error during vector index retrieval: {e}")
            raise
    def bm25_retriever(self, query: str) -> List[NodeWithScore]:
        try:
            rprint("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
            rprint("[bold cyan]            2. BM25 Retriever                      [/bold cyan]")
            rprint("[bold cyan]" + "=" * 60 + "[/bold cyan]")
            
            # Custom parameterization values
            k1_val = 1.2
            b_val = 0.4
            
            import bm25s
            from llama_index.retrievers.bm25.base import node_to_metadata_dict
            
            try:
                import Stemmer
                stemmer = Stemmer.Stemmer("english")
                skip_stemming = False
            except ImportError:
                logger.warning("Stemmer package missing. Running BM25 without stemming.")
                stemmer = None
                skip_stemming = True
            
            # 1. Instantiate the BM25 model with our custom parameters
            bm25_model = bm25s.BM25(k1=k1_val, b=b_val)
            
            # 2. Tokenize the corpus nodes using LlamaIndex standards
            corpus_tokens = bm25s.tokenize(
                [node.get_content() for node in self.nodes],
                stopwords="english",
                stemmer=stemmer if not skip_stemming else None
            )
            
            # 3. Build the scoring index matrix
            bm25_model.index(corpus_tokens, show_progress=False)
            
            # 4. Map LlamaIndex nodes metadata to the corpus structure
            bm25_model.corpus = [
                node_to_metadata_dict(node) | {"node_id": node.node_id}
                for node in self.nodes
            ]
            
            # 5. Create the retriever with the parameterized model
            bm25_retriever = BM25Retriever(
                existing_bm25=bm25_model,
                similarity_top_k=3,
                stemmer=stemmer,
                skip_stemming=skip_stemming
            )
            
            return bm25_retriever.retrieve(query)
        except Exception as e:
            logger.warning(f"BM25 retrieval failed or error occurred ({e}). Falling back to Vector Search...")
            return self.vector_index_retriever(query)

            
if __name__ == "__main__":
    lab = AdvanceRetrieversLab()
    query = lab.demo_queries["basic"]
    response = lab.vector_index_retriever(query)
    
    rprint(f"\n[bold yellow]🔍 Query:[/bold yellow] [italic]{query}[/italic]")
    rprint(f"[bold green]✨ Retrieved {len(response)} nodes:[/bold green]")
    for i, node in enumerate(response, 1):
        score = node.score if node.score is not None else 0.0
        rprint(f"\n[bold cyan]Match {i}[/bold cyan] (Score: [bold green]{score:.4f}[/bold green])")
        rprint(f"  [dim]{node.text}[/dim]")
    
    query = lab.demo_queries['technical']
    response = lab.bm25_retriever(query)
    
    rprint(f"\n[bold yellow]🔍 Query (BM25 Keyword):[/bold yellow] [italic]{query}[/italic]")
    rprint("[bold white]BM25 analyzes exact keyword matches with sophisticated scoring[/bold white]")
    rprint(f"[bold green]✨ Retrieved {len(response)} nodes:[/bold green]")
    
    for i, node in enumerate(response, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0.0
        rprint(f"\n[bold cyan]Match {i}[/bold cyan] (BM25 Score: [bold green]{score:.4f}[/bold green])")
        rprint(f"  [dim]{node.text}[/dim]")
        
        # Highlight which query terms appear in the text
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        if found_terms:
            rprint(f"   [yellow]→ Found terms:[/yellow] {found_terms}")
            
    # Print the BM25 vs TF-IDF explanation only once, outside the loop
    rprint("\n[bold cyan]🧠 BM25 vs TF-IDF Comparison Summary:[/bold cyan]")
    rprint("[bold white]1. TF-IDF Problem:[/bold white] Linear term frequency scaling (e.g., 100 occurrences → 100x score)")
    rprint("   [bold green]BM25 Solution:[/bold green] Saturation function (scores plateau as occurrences increase)")
    rprint("[bold white]2. TF-IDF Problem:[/bold white] No document length consideration (long documents dominate)")
    rprint("   [bold green]BM25 Solution:[/bold green] Length normalization (b parameter adjusts score based on document length relative to average)")
    rprint("[bold white]3. Key BM25 Parameters:[/bold white]")
    rprint("   - [italic]k1[/italic] ≈ 1.2: Controls term frequency saturation plateau")
    rprint("   - [italic]b[/italic] ≈ 0.75: Controls document length normalization (0=none, 1=full)")
    rprint("   - [italic]IDF weighting[/italic]: Rare terms automatically get higher relevance scores")
        