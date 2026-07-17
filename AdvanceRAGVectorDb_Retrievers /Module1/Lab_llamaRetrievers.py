from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
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
from explanations import BACKGROUND_HELP, VECTOR_INDEX_HELP, BM25_HELP, AUTO_MERGING_RETRIEVER

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

    # Background details and index types overview have been moved to explanations.py (BACKGROUND_HELP)

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
        """Vector Index Retriever - The Foundation (See explanations.VECTOR_INDEX_HELP)."""
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
        """BM25 Retriever - Advanced Keyword Search (See explanations.BM25_HELP)."""
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
    def document_summary_retriever(self, query: str) -> tuple[List[NodeWithScore], List[NodeWithScore]]:
        """DOCUMENT_SUMMARY Retriever - Advanced Keyword Search (See explanations.DOCUMENT_SUMMARY_HELP)."""
        try:
            rprint("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
            rprint("[bold cyan]            4. DOCUMENT SUMMARY INDEX RETRIEVERS                     [/bold cyan]")
            rprint("[bold cyan]" + "=" * 60 + "[/bold cyan]")
            # LLM-based document summary retriever
            doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
                self.document_summary_index,
                choice_top_k=3
            )
            # Embedding-based document summary retriever 
            doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
                self.document_summary_index,
                similarity_top_k=3  
            )
            llm_results = doc_summary_retriever_llm.retrieve(query)
            embed_results = doc_summary_retriever_embedding.retrieve(query)

            return llm_results, embed_results
        except Exception as e:
            logger.warning(f"DOCUMENT SUMMARY INDEX retrieval failed or error occurred ({e}). Falling back to Vector Search...")
            fallback_res = self.vector_index_retriever(query)
            return fallback_res, fallback_res
    def auto_merging_retriever(self,query:str)-> List[NodeWithScore]:
        """AUTO MERGING RETRIEVER Retriever - Advanced Keyword Search (See explanations.AUTO_MERGING_RETRIEVER)."""
        try:
            rprint("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
            rprint("[bold cyan]            4. AUTO MERGING RETRIEVER RETRIEVERS                     [/bold cyan]")
            rprint("[bold cyan]" + "=" * 60 + "[/bold cyan]")
            # Create hierarchical nodes
            node_parser=HierarchicalNodeParser.from_defaults(
                chunk_sizes=[512,256,128]
            )
            hier_nodes= node_parser.get_nodes_from_documents(self.documents)
            # Create storage context with all nodes
            docstore= SimpleDocumentStore()
            docstore.add_documents(hier_nodes)
            storage_context=StorageContext.from_defaults(
                docstore=docstore
            )
            # Create base index
            base_index=VectorStoreIndex(hier_nodes,storage_context)
            base_retriever=base_index.as_retriever(similarity_top_k=6)
            # Create auto-merging retriever
            auto_merging_retriever = AutoMergingRetriever(
                base_retriever,
                storage_context,
                verbose=True
            )
            llm_results = auto_merging_retriever.retrieve(query)
            return llm_results
        except Exception as e:
            logger.warning(f"AUTO MERGING retrieval failed or error occurred ({e}). Falling back to Vector Search...")
            fallback_res = self.vector_index_retriever(query)
            return fallback_res

# =====================================================================
# PRESENTATION & DISPLAY HELPER FUNCTIONS (Senior Dev Best Practice)
# =====================================================================

def display_retrieval_results(query: str, response: List[NodeWithScore], retriever_name: str, is_bm25: bool = False) -> None:
    """Formats and prints standard vector or keyword-based search results."""
    rprint(f"\n[bold yellow]🔍 Query ({retriever_name}):[/bold yellow] [italic]{query}[/italic]")
    if is_bm25:
        rprint("[bold white]BM25 analyzes exact keyword matches with sophisticated scoring[/bold white]")
    rprint(f"[bold green]✨ Retrieved {len(response)} nodes:[/bold green]")
    
    for i, node in enumerate(response, 1):
        score = node.score if getattr(node, 'score', None) is not None else 0.0
        score_label = "BM25 Score" if is_bm25 else "Score"
        rprint(f"\n[bold cyan]Match {i}[/bold cyan] ({score_label}: [bold green]{score:.4f}[/bold green])")
        rprint(f"  [dim]{node.text}[/dim]")
        
        if is_bm25:
            text_lower = node.text.lower()
            query_terms = query.lower().split()
            found_terms = [term for term in query_terms if term in text_lower]
            if found_terms:
                rprint(f"   [yellow]→ Found terms:[/yellow] {found_terms}")


def display_summary_retrieval_results(query: str, llm_response: List[NodeWithScore], embed_response: List[NodeWithScore]) -> None:
    """Formats and prints comparative document summary retriever results."""
    rprint(f"\n[bold yellow]🔍 Query (Summary):[/bold yellow] [italic]{query}[/italic]")
    
    rprint("\n[bold green]✨ A) LLM-based Document Summary Retriever:[/bold green]")
    rprint("[dim]Uses LLM reasoning to evaluate document summaries[/dim]")
    for i, node in enumerate(llm_response, 1):
        score_str = f"{node.score:.1f}" if getattr(node, "score", None) is not None else "N/A"
        rprint(f"  {i}. [bold cyan]Relevance Rating:[/bold cyan] {score_str}")
        rprint(f"     [dim]{node.text[:120]}...[/dim]")
        
    rprint("\n[bold green]✨ B) Embedding-based Document Summary Retriever:[/bold green]")
    rprint("[dim]Uses vector similarity against summary embeddings[/dim]")
    for i, node in enumerate(embed_response, 1):
        score_str = f"{node.score:.4f}" if getattr(node, "score", None) is not None else "N/A (Not propagated)"
        rprint(f"  {i}. [bold cyan]Similarity Score:[/bold cyan] {score_str}")
        rprint(f"     [dim]{node.text[:120]}...[/dim]")


def display_bm25_explanation() -> None:
    """Prints a clean conceptual overview comparing BM25 vs TF-IDF."""
    rprint("\n[bold cyan]🧠 BM25 vs TF-IDF Comparison Summary:[/bold cyan]")
    rprint("[bold white]1. TF-IDF Problem:[/bold white] Linear term frequency scaling (e.g., 100 occurrences → 100x score)")
    rprint("   [bold green]BM25 Solution:[/bold green] Saturation function (scores plateau as occurrences increase)")
    rprint("[bold white]2. TF-IDF Problem:[/bold white] No document length consideration (long documents dominate)")
    rprint("   [bold green]BM25 Solution:[/bold green] Length normalization (b parameter adjusts score based on document length relative to average)")
    rprint("[bold white]3. Key BM25 Parameters:[/bold white]")
    rprint("   - [italic]k1[/italic] ≈ 1.2: Controls term frequency saturation plateau")
    rprint("   - [italic]b[/italic] ≈ 0.75: Controls document length normalization (0=none, 1=full)")
    rprint("   - [italic]IDF weighting[/italic]: Rare terms automatically get higher relevance scores")


def display_auto_merging_results(query: str, response: List[NodeWithScore]) -> None:
    """Formats and prints auto-merging retriever results, showing if nodes were auto-merged."""
    rprint(f"\n[bold yellow]🔍 Query (Auto Merging):[/bold yellow] [italic]{query}[/italic]")
    rprint(f"[bold green]✨ Retrieved {len(response)} nodes:[/bold green]")
    for i, node in enumerate(response[:3], 1):
        score = node.score if getattr(node, 'score', None) is not None else 0.0
        score_str = f"{score:.4f}" if score > 0.0 else "N/A (Auto-merged/Parent context)"
        rprint(f"  {i}. [bold cyan]Score:[/bold cyan] {score_str}")
        rprint(f"     [dim]{node.text[:120]}...[/dim]")

def display_auto_merging_explanation() -> None:
    """Prints a clean conceptual overview of Auto Merging Retriever context preservation."""
    rprint("\n[bold cyan]🧠 Auto Merging Retriever Context Summary:[/bold cyan]")
    rprint(AUTO_MERGING_RETRIEVER.strip())


if __name__ == "__main__":
    lab = AdvanceRetrieversLab()
    
    # 1. Vector Index Retriever
    query_basic = lab.demo_queries["basic"]
    vector_res = lab.vector_index_retriever(query_basic)
    display_retrieval_results(query_basic, vector_res, "Vector")
    
    # 2. BM25 Retriever
    query_tech = lab.demo_queries['technical']
    bm25_res = lab.bm25_retriever(query_tech)
    display_retrieval_results(query_tech, bm25_res, "BM25 Keyword", is_bm25=True)
    display_bm25_explanation()
    
    # 3. Document Summary Index Retrievers
    query_sum = lab.demo_queries['learning_types']
    llm_res, embed_res = lab.document_summary_retriever(query_sum)
    display_summary_retrieval_results(query_sum, llm_res, embed_res)
    
    # 4. Auto Merging Retrievers
    query_merging = lab.demo_queries['advanced']
    merging_resp = lab.auto_merging_retriever(query_merging)
    display_auto_merging_results(query_merging, merging_resp)
    display_auto_merging_explanation()