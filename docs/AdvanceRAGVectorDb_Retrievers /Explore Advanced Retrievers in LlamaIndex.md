<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
    </a>
</p>


# **Explore Advanced Retrievers in LlamaIndex**

Estimated time needed: **60** minutes

This comprehensive lab demonstrates advanced retrieval techniques in LlamaIndex using IBM watsonx.ai as the foundation model provider. You'll learn core retrievers, advanced retrievers, and sophisticated fusion techniques that power modern RAG applications.

Through hands-on examples, you'll master the art of building intelligent information retrieval systems that can handle complex queries, combine multiple search strategies, and deliver precise results for production RAG applications.

## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
            <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
            <li><a href="#watsonx.ai-LLM-Integration">watsonx.ai LLM Integration</a></li>
            <li><a href="#Sample-Data-Setup">Sample Data Setup</a></li>
        </ol>
    </li>
    <li>
        <a href="#Background">Background</a>
        <ol>
            <li><a href="#What-are-Advanced-Retrievers?">What are Advanced Retrievers?</a></li>
            <li><a href="#Why-are-Advanced-Retrievers-Important?">Why are Advanced Retrievers Important?</a></li>
            <li><a href="#Index-Types-Overview">Index Types Overview</a></li>
        </ol>
    </li>
    <li>
        <a href="#Core-Retriever-Demonstrations">Core Retriever Demonstrations</a>
        <ol>
            <li><a href="#1.-Vector-Index-Retriever---The-Foundation">Vector Index Retriever - The Foundation</a></li>
            <li><a href="#2.-BM25-Retriever---Advanced-Keyword-Based-Search">BM25 Retriever - Advanced Keyword Search</a></li>
            <li><a href="#3.-Document-Summary-Index-Retrievers">Document Summary Index Retrievers</a></li>
            <li><a href="#4.-Auto-Merging-Retriever---Hierarchical-Context-Preservation">Auto Merging Retriever - Hierarchical Context</a></li>
            <li><a href="#5.-Recursive-Retriever---Multi-Level-Reference-Following">Recursive Retriever - Multi-Level Reference Following</a></li>
            <li><a href="#6.-Query-Fusion-Retriever---Multi-Query-Enhancement-with-Advanced-Fusion">QueryFusion Retriever - Multi-Query Enhancement</a></li>
        </ol>
    </li>
    <li>
        <a href="#Exercises">Exercises</a>
        <ol>
            <li><a href="#Exercise-1---Build-a-Custom-Hybrid-Retriever">Exercise 1 - Build a Custom Hybrid Retriever</a></li>
            <li><a href="#Exercise-2---Create-a-Production-RAG-Pipeline">Exercise 2 - Create a Production RAG Pipeline</a></li>
        </ol>
    </li>
    <li><a href="#Summary">Summary</a></li>
    <li><a href="#Authors">Authors</a></li>
</ol>


## Objectives

After completing this lab you will be able to:

- Understand the different types of retrievers available in LlamaIndex and their use cases
- Implement Vector Index Retriever for semantic search
- Use BM25 Retriever for keyword-based search with advanced ranking
- Create Document Summary Index Retrievers for intelligent document selection
- Build Auto Merging Retriever for hierarchical context preservation
- Implement Recursive Retriever for multi-level reference following
- Master QueryFusion Retriever with advanced fusion techniques (RRF, Relative Score, Distribution-Based)
- Compare and contrast different retrieval approaches for various scenarios
- Build production-ready RAG pipelines with multiple retrieval strategies

---


## Setup

For this lab, we will be using the following libraries:

*   [`llama-index`](https://docs.llamaindex.ai/) - The core LlamaIndex library for building RAG applications
*   [`llama-index-llms-ibm`](https://docs.llamaindex.ai/en/stable/api_reference/llms/ibm/) - IBM watsonx.ai integration for LlamaIndex
*   [`llama-index-retrievers-bm25`](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/bm25/) - BM25 retriever implementation
*   [`llama-index-embeddings-huggingface`](https://docs.llamaindex.ai/en/stable/api_reference/embeddings/huggingface/) - HuggingFace embeddings integration
*   [`sentence-transformers`](https://www.sbert.net/) - For generating high-quality text embeddings
*   [`rank-bm25`](https://github.com/dorianbrown/rank_bm25) - BM25 ranking algorithm implementation
*   [`PyStemmer`](https://github.com/snowballstem/pystemmer) - Stemming algorithms for better text processing
*   [`ibm-watsonx-ai`](https://ibm.github.io/watsonx-ai-python-sdk/) - IBM watsonx.ai SDK for foundation models


### Installing Required Libraries

Run the following cell to install required libraries.

**NOTE**: The installation process takes about **5** minutes to complete. Feel free to grab a coffee in the meantime

```
  ( (
   ) )
........
|      |]
\      /   
 `----'
```


```python
!pip install llama-index==0.12.49 \
    llama-index-embeddings-huggingface==0.5.5 \
    llama-index-llms-ibm==0.4.0 \
    llama-index-retrievers-bm25==0.5.2 \
    sentence-transformers==5.0.0 \
    rank-bm25==0.2.2 \
    PyStemmer==2.2.0.3 \
    ibm-watsonx-ai==1.3.31 | tail -n 1
```

### Importing Required Libraries

We import all the necessary libraries for this lab, including core LlamaIndex components, retrievers, and IBM watsonx.ai integration:


```python
import os
import json
from typing import List, Optional
import asyncio
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.retrievers import (
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

# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

# IBM WatsonX LlamaIndex integration
from ibm_watsonx_ai import APIClient
from llama_index.llms.ibm import WatsonxLLM

# Sentence transformers
from sentence_transformers import SentenceTransformer

# Statistical libraries for fusion techniques
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some advanced fusion features will be limited")

print("✅ All imports successful!")
```

## watsonx.ai LLM Integration

We'll create custom wrapper classes to integrate IBM watsonx.ai with LlamaIndex. This allows us to use watsonx.ai foundation models while maintaining compatibility with all LlamaIndex retrievers.


```python
# watsonx.ai LLM using official LlamaIndex integration
def create_watsonx_llm():
    """Create watsonx.ai LLM instance using official LlamaIndex integration."""
    try:
        # Create the API client object
        api_client = APIClient({'url': "https://us-south.ml.cloud.ibm.com"})
        # Use llama-index-llms-ibm (official watsonx.ai integration)
        llm = WatsonxLLM(
            model_id="ibm/granite-4-h-small",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            api_client=api_client,
            temperature=0.9
        )
        print("✅ watsonx.ai LLM initialized using official LlamaIndex integration")
        return llm
    except Exception as e:
        print(f"⚠️ watsonx.ai initialization error: {e}")
        print("Falling back to mock LLM for demonstration")
        
        # Fallback mock LLM for demonstration
        from llama_index.core.llms.mock import MockLLM
        return MockLLM(max_tokens=512)
```

```python
# Initialize embedding model first
print("🔧 Initializing HuggingFace embeddings...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
print("✅ HuggingFace embeddings initialized!")

# Setup with watsonx.ai
print("🔧 Initializing watsonx.ai LLM...")
llm = create_watsonx_llm()

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model
print("✅ watsonx.ai LLM and embeddings configured!")
```

---

## Background

Before diving into the advanced retrieval techniques, let's understand the foundational concepts that make these retrievers powerful.

### What are Advanced Retrievers?

Advanced retrievers in LlamaIndex are sophisticated components that go beyond simple vector similarity search to provide more nuanced, context-aware, and intelligent information retrieval. They combine multiple techniques such as:

- **Semantic Understanding**: Using embeddings to understand meaning and context
- **Keyword Matching**: Precise term-based search for exact specifications
- **Hierarchical Context**: Maintaining relationships between different levels of information
- **Multi-Query Processing**: Generating and combining results from multiple query variations
- **Fusion Techniques**: Intelligently combining results from different retrieval methods

### Why are Advanced Retrievers Important?

1. **Improved Accuracy**: Advanced retrievers can find more relevant information by using multiple search strategies
2. **Better Context Preservation**: They maintain important relationships between pieces of information
3. **Reduced Hallucination**: More precise retrieval leads to more accurate AI responses
4. **Scalability**: Efficient retrieval strategies work better with large document collections
5. **Flexibility**: Different retrieval methods can be combined for optimal results

### Index Types Overview

Before exploring advanced retrievers, it's helpful to first understand the three main index types supported by LlamaIndex. Each is designed to support different retrieval scenarios:

**VectorStoreIndex:**
- Stores vector embeddings for each document chunk
- Best suited for semantic retrieval based on meaning
- Commonly used in LLM pipelines and RAG applications

**DocumentSummaryIndex:**
- Generates and stores summaries of documents at indexing time
- Uses summaries to filter documents before retrieving full content
- Especially useful for large and diverse document sets that cannot fit in the context window of an LLM

**KeywordTableIndex:**
- Extracts keywords from documents and maps them to specific content chunks
- Enables exact keyword matching for rule-based or hybrid search scenarios
- Ideal for applications requiring precise term matching

## Sample Data Setup

We'll use a collection of AI and machine learning documents to demonstrate different retrieval strategies.


```python
# Sample data for the lab - AI/ML focused documents
SAMPLE_DOCUMENTS = [
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
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning", 
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}

print(f"📄 Loaded {len(SAMPLE_DOCUMENTS)} sample documents")
print(f"🔍 Prepared {len(DEMO_QUERIES)} consistent demo queries")
for i, doc in enumerate(SAMPLE_DOCUMENTS[:3], 1):
    print(f"{i}. {doc}")
print("...")
```

## Initialize Lab Environment

Let's create our lab class and initialize all the indexes we'll need for different retrievers.


```python
class AdvancedRetrieversLab:
    def __init__(self):
        print("🚀 Initializing Advanced Retrievers Lab...")
        self.documents = [Document(text=text) for text in SAMPLE_DOCUMENTS]
        self.nodes = SentenceSplitter().get_nodes_from_documents(self.documents)
        
        print("📊 Creating indexes...")
        # Create various indexes
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        self.document_summary_index = DocumentSummaryIndex.from_documents(self.documents)
        self.keyword_index = KeywordTableIndex.from_documents(self.documents)
        
        print("✅ Advanced Retrievers Lab Initialized!")
        print(f"📄 Loaded {len(self.documents)} documents")
        print(f"🔢 Created {len(self.nodes)} nodes")

# Initialize the lab
lab = AdvancedRetrieversLab()
```

---

# Core Retriever Demonstrations

This lab focuses on the essential retrievers in LlamaIndex, covering core retrieval methods, advanced retrievers, and fusion techniques. Each section provides practical examples and detailed explanations based on official LlamaIndex documentation.


## 1. Vector Index Retriever - The Foundation

The Vector Index Retriever uses vector embeddings to find semantically related content, making it ideal for general-purpose search and widely used in retrieval-augmented generation (RAG) pipelines.

**How it works**: 
- Documents are split into nodes and embedded using the configured embedding model
- Query is converted to an embedding vector
- Returns nodes ranked by cosine similarity to the query embedding
- Generates embeddings in batches of 2048 nodes by default

**When to use:**
- General-purpose semantic search (most common use case)
- Finding conceptually related content based on meaning rather than exact keywords
- RAG pipelines where semantic understanding is crucial
- When exact keyword matching isn't the primary requirement

**Key characteristics from authoritative source:**
- **Stores embeddings for each document chunk** (VectorStoreIndex foundation)
- **Best for semantic retrieval** based on meaning and context
- **Commonly used in LLM pipelines** for retrieval-augmented generation

**Strengths**: 
- Excellent semantic understanding and context awareness
- Handles synonyms and related concepts effectively
- Works well with natural language queries

**Limitations**: 
- May miss exact keyword matches when specific terms are crucial
- Requires a good embedding model for optimal performance
- Can be computationally intensive for large document collections


```python
print("=" * 60)
print("1. VECTOR INDEX RETRIEVER")
print("=" * 60)

# Basic vector retriever
vector_retriever = VectorIndexRetriever(
    index=lab.vector_index,
    similarity_top_k=3
)

# Alternative creation method
alt_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["basic"]  # "What is machine learning?"
nodes = vector_retriever.retrieve(query)

print(f"Query: {query}")
print(f"Retrieved {len(nodes)} nodes:")
for i, node in enumerate(nodes, 1):
    print(f"{i}. Score: {node.score:.4f}")
    print(f"   Text: {node.text[:100]}...")
    print()
```

## 2. BM25 Retriever - Advanced Keyword-Based Search

BM25 is a keyword-based retrieval method that improves on TF-IDF by addressing some of its key limitations. It's widely used in production search systems including Elasticsearch and Apache Lucene.

### Understanding TF-IDF: The Foundation

Before diving into BM25, let's understand **TF-IDF** (Term Frequency-Inverse Document Frequency), which BM25 builds upon:

**Term Frequency (TF)**: Measures how often a word appears in a document
- Example: If "neural" appears 3 times in a 100-word document, TF = 3/100 = 0.03

**Inverse Document Frequency (IDF)**: Measures how rare a word is across all documents
- Example: If "neural" appears in only 2 out of 1000 documents, IDF = log(1000/2) = 6.21
- Common words like "the" have low IDF; rare technical terms have high IDF

**TF-IDF Score**: TF × IDF
- Highlights words that are frequent in one document but rare across the collection
- Developed by Karen Spärck Jones, who pioneered the concept of term specificity

### How BM25 Improves Upon TF-IDF

**Key BM25 Improvements:**

1. **Term Frequency Saturation**: BM25 reduces the impact of repeated terms using term frequency saturation
   - Problem: In TF-IDF, if a word appears 100 times vs 10 times, the score increases linearly
   - Solution: BM25 uses a saturation function that plateaus after a certain frequency

2. **Document Length Normalization**: BM25 adjusts for document length, making it more effective for keyword-based search
   - Problem: In TF-IDF, longer documents have unfair advantages
   - Solution: BM25 normalizes scores based on document length relative to average

3. **Tunable Parameters**: Allows fine-tuning for different types of content
   - k1 ≈ 1.2: Controls term frequency saturation (how quickly scores plateau)
   - b ≈ 0.75: Controls document length normalization (0=none, 1=full)

### When to Use BM25

**Ideal for:**
- Technical documentation where exact terms matter
- Legal documents with specific terminology
- Product catalogs with precise specifications
- Academic papers with specialized vocabulary
- Applications requiring keyword-based retrieval rather than semantic similarity

**Advantages:**
- Excellent precision for exact term matches
- Fast computational performance
- Proven effectiveness in production systems
- No training required (unlike neural approaches)
- Interpretable scoring mechanism

**Limitations:**
- No semantic understanding (doesn't handle synonyms)
- Struggles with typos and variations
- Limited context understanding
- Requires careful parameter tuning for optimal performance


```python
print("=" * 60)
print("2. BM25 RETRIEVER")
print("=" * 60)

try:
    import Stemmer
    
    # Create BM25 retriever with default parameters
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=lab.nodes,
        similarity_top_k=3,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )
    
    query = DEMO_QUERIES["technical"]  # "neural networks deep learning"
    nodes = bm25_retriever.retrieve(query)
    
    print(f"Query: {query}")
    print("BM25 analyzes exact keyword matches with sophisticated scoring")
    print(f"Retrieved {len(nodes)} nodes:")
    
    for i, node in enumerate(nodes, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0
        print(f"{i}. BM25 Score: {score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        
        # Highlight which query terms appear in the text
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        if found_terms:
            print(f"   → Found terms: {found_terms}")
        print()
    
    print("BM25 vs TF-IDF Comparison:")
    print("TF-IDF Problem: Linear term frequency scaling")
    print("  Example: 10 occurrences → score of 10, 100 occurrences → score of 100")
    print("BM25 Solution: Saturation function")
    print("  Example: 10 occurrences → high score, 100 occurrences → slightly higher score")
    print()
    print("TF-IDF Problem: No document length consideration")
    print("  Example: Long documents dominate results")
    print("BM25 Solution: Length normalization (b parameter)")
    print("  Example: Scores adjusted based on document length vs. average")
    print()
    print("Key BM25 Parameters:")
    print("- k1 ≈ 1.2: Term frequency saturation (how quickly scores plateau)")
    print("- b ≈ 0.75: Document length normalization (0=none, 1=full)")
    print("- IDF weighting: Rare terms get higher scores")
        
except ImportError:
    print("⚠️ BM25Retriever requires 'pip install PyStemmer'")
    print("Demonstrating BM25 concepts with fallback vector search...")
    
    fallback_retriever = lab.vector_index.as_retriever(similarity_top_k=3)
    query = DEMO_QUERIES["technical"]
    nodes = fallback_retriever.retrieve(query)
    
    print(f"Query: {query}")
    print("(Using vector fallback to demonstrate BM25 concepts)")
    
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Vector Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        
        # Demonstrate TF-IDF concept manually
        text_lower = node.text.lower()
        query_terms = query.lower().split()
        found_terms = [term for term in query_terms if term in text_lower]
        
        if found_terms:
            print(f"   → BM25 would boost this result for terms: {found_terms}")
        print()
    
    print("BM25 Concept Demonstration:")
    print("1. TF-IDF Foundation:")
    print("   - Term Frequency: How often words appear in document")
    print("   - Inverse Document Frequency: How rare words are across collection")
    print("   - TF-IDF = TF × IDF (balances frequency vs rarity)")
    print()
    print("2. BM25 Improvements:")
    print("   - Saturation: Prevents over-scoring repeated terms")
    print("   - Length normalization: Prevents long document bias")
    print("   - Tunable parameters: k1 (saturation) and b (length adjustment)")
    print()
    print("3. Real-world Usage:")
    print("   - Elasticsearch default scoring function")
    print("   - Apache Lucene/Solr standard")
    print("   - Used in 83% of text-based recommender systems")
    print("   - Developed by Robertson & Spärck Jones at City University London")
```

## 3. Document Summary Index Retrievers

Document Summary Index Retrievers use document summaries instead of the actual documents to find relevant content, making them efficient for large collections. **They return the original documents, not their summaries.**

**How it works (from authoritative source)**:
- **Generates and stores summaries of documents** at indexing time
- **Uses summaries to filter documents** before retrieving full content
- **Two-stage Process**: First uses summaries to filter documents, then returns full document content
- **Especially useful for large, diverse corpora** that cannot fit in the context window of an LLM

**Two Retrieval Options**: 
1. **DocumentSummaryIndexLLMRetriever**: 
   - Uses a large language model to analyze the query against document summaries
   - Provides intelligent document selection but can be more time-consuming and expensive
   - Best for complex queries requiring nuanced understanding

2. **DocumentSummaryIndexEmbeddingRetriever**: 
   - Uses semantic similarity between the query and summary embeddings
   - Faster and more cost-effective than LLM-based approach
   - Good for straightforward similarity matching

**When to use (based on authoritative guidance):**
- Large document collections where documents cover different topics
- When you need efficient document-level filtering before detailed retrieval
- Multi-document QA where documents have distinct subject matters
- Large and diverse document sets that cannot fit in the context window of an LLM

**Configuration Parameters:**
- `choice_top_k` (LLM retriever): Number of documents to select
- `similarity_top_k` (Embedding retriever): Number of documents to select
- Default is 1, increase for multiple document retrieval

**Key Point**: **Returns original documents, not their summaries** - the summaries are only used for filtering

**Strengths**: 
- Efficient document selection and reduces search space
- Good for heterogeneous collections with diverse topics
- Returns original documents with full context intact

**Limitations**: 
- Requires LLM for summary generation during indexing
- May lose some detail present in original documents during summary creation
- LLM-based version can be slower and more expensive than other options


```python
print("=" * 60)
print("3. DOCUMENT SUMMARY INDEX RETRIEVERS")
print("=" * 60)

# LLM-based document summary retriever
doc_summary_retriever_llm = DocumentSummaryIndexLLMRetriever(
    lab.document_summary_index,
    choice_top_k=3  # Number of documents to select
)

# Embedding-based document summary retriever  
doc_summary_retriever_embedding = DocumentSummaryIndexEmbeddingRetriever(
    lab.document_summary_index,
    similarity_top_k=3  # Number of documents to select
)

query = DEMO_QUERIES["learning_types"]  # "different types of learning"

print(f"Query: {query}")

print("\nA) LLM-based Document Summary Retriever:")
print("Uses LLM to select relevant documents based on summaries")
try:
    nodes_llm = doc_summary_retriever_llm.retrieve(query)
    print(f"Retrieved {len(nodes_llm)} nodes")
    for i, node in enumerate(nodes_llm[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f"   Text: {node.text[:80]}...")
        print()
except Exception as e:
    print(f"LLM-based retrieval demo: {str(e)[:100]}...")

print("B) Embedding-based Document Summary Retriever:")
print("Uses vector similarity between query and document summaries")
try:
    nodes_emb = doc_summary_retriever_embedding.retrieve(query)
    print(f"Retrieved {len(nodes_emb)} nodes")
    for i, node in enumerate(nodes_emb[:2], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Document summary)")
        print(f"   Text: {node.text[:80]}...")
        print()
except Exception as e:
    print(f"Embedding-based retrieval demo: {str(e)[:100]}...")

print("Document Summary Index workflow:")
print("1. Generates summaries for each document using LLM")
print("2. Uses summaries to select relevant documents")
print("3. Returns full content from selected documents")
```

## 4. Auto Merging Retriever - Hierarchical Context Preservation

Auto Merging Retriever is designed to preserve context in long documents using a hierarchical structure. **It uses hierarchical chunking to break documents into parent and child nodes, and if enough child nodes from the same parent are retrieved, the retriever returns the parent node instead.**

**How it works (from authoritative source)**:
- **Uses hierarchical chunking** to break documents into parent and child nodes
- **Retrieves parent if enough children match** - intelligent merging logic
- **Preserves context in long documents** by consolidating related content
- **Dual Storage**: Smaller child chunks are indexed in the vector store for precise matching, while larger parent chunks are stored in the docstore

**Key behavior pattern**:
- Child chunks enable precise matching for specific queries
- When multiple child chunks from the same parent are retrieved, the system returns the parent chunk
- This **helps consolidate related content and preserve broader context**

**When to use (based on authoritative guidance):**
- Long documents where small chunks lose important surrounding context
- Legal documents, research papers, technical specifications that need context preservation
- When you need both precise matching and comprehensive context
- Documents with natural hierarchical structure (sections, subsections)

**Configuration:**
- `chunk_sizes`: List of chunk sizes from largest to smallest (e.g., [512, 256, 128])
- `chunk_overlap`: Overlap between chunks to maintain continuity
- Storage context manages both vector store (child nodes) and docstore (parent nodes)

**Strengths**: 
- Automatically preserves context without manual intervention
- Reduces information fragmentation in long documents
- Intelligent merging based on retrieval patterns
- Maintains granular search capability while providing broader context

**Limitations**: 
- More complex setup compared to basic retrievers
- Requires hierarchical document structure to be effective
- Higher storage overhead due to multiple chunk levels
- May not be suitable for very short documents

*Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/*


```python
print("=" * 60)
print("4. AUTO MERGING RETRIEVER")
print("=" * 60)

# Create hierarchical nodes
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 256, 128]
)

hier_nodes = node_parser.get_nodes_from_documents(lab.documents)

# Create storage context with all nodes
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore

docstore = SimpleDocumentStore()
docstore.add_documents(hier_nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

# Create base index
base_index = VectorStoreIndex(hier_nodes, storage_context=storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=6)

# Create auto-merging retriever
auto_merging_retriever = AutoMergingRetriever(
    base_retriever, 
    storage_context,
    verbose=True
)

query = DEMO_QUERIES["advanced"]  # "How do neural networks work in deep learning?"
nodes = auto_merging_retriever.retrieve(query)

print(f"Query: {query}")
print(f"Auto-merged to {len(nodes)} nodes")
for i, node in enumerate(nodes[:3], 1):
    print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Auto-merged)")
    print(f"   Text: {node.text[:120]}...")
    print()
```

## 5. Recursive Retriever - Multi-Level Reference Following

The Recursive Retriever is **designed to follow relationships between nodes using references**. **It can follow references from one node to another, such as citations in academic papers or other metadata links**, allowing it to **retrieve related content across documents or layers of abstraction**.

**How it works (from authoritative source)**:
- **Follows node references** - traverses relationships to find referenced content
- **Supports chunk and metadata linking** - handles different types of references
- **Multi-Level Navigation**: Can execute sub-queries on referenced retrievers or query engines
- **Network Building**: Creates a network of interconnected retrievers that can reference each other

**Reference Types Supported**:
1. **Chunk References**: Smaller child chunks refer to larger parent chunks for additional context
2. **Metadata References**: Summaries or generated questions refer to larger content chunks, such as citations in academic papers

**When to use (based on authoritative guidance):**
- **Academic papers with citations** and extensive references
- **Research papers** where you need to retrieve relevant content from cited papers
- Documentation with cross-references and linked content
- Knowledge bases with interconnected information
- When nodes reference structured data (tables, databases, other documents)

**Configuration:**
- `retriever_dict`: Maps node IDs or keys to specific retrievers
- `query_engine_dict`: Maps keys to query engines for sub-queries
- Node metadata can contain references to other nodes or data structures

**Key capability**: **Retrieves related content across documents** by following reference chains

**Strengths**: 
- Follows complex relationships and enables multi-step reasoning
- Provides comprehensive coverage across related documents
- Excellent for handling interconnected information systems
- Can traverse multiple levels of references automatically

**Limitations**: 
- Requires careful setup of node relationships
- Can be computationally expensive for deep reference chains
- Complex debugging when reference chains are extensive
- May retrieve too much related content if not properly configured

*Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/recurisve_retriever_nodes_braintrust/*


```python
print("=" * 60)
print("5. RECURSIVE RETRIEVER")
print("=" * 60)

# Create documents with references
docs_with_refs = []
for i, doc in enumerate(lab.documents):
    # Add reference metadata
    ref_doc = Document(
        text=doc.text,
        metadata={
            "doc_id": f"doc_{i}",
            "references": [f"doc_{j}" for j in range(len(lab.documents)) if j != i][:2]
        }
    )
    docs_with_refs.append(ref_doc)

# Create index with referenced documents
ref_index = VectorStoreIndex.from_documents(docs_with_refs)

# Create retriever mapping
retriever_dict = {
    f"doc_{i}": ref_index.as_retriever(similarity_top_k=1)
    for i in range(len(docs_with_refs))
}

# Base retriever
base_retriever = ref_index.as_retriever(similarity_top_k=2)

# Add the root retriever to the dictionary
retriever_dict["vector"] = base_retriever

# Recursive retriever
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict=retriever_dict,
    query_engine_dict={},
    verbose=True
)

query = DEMO_QUERIES["applications"]  # "What are the applications of AI?"
try:
    nodes = recursive_retriever.retrieve(query)
    print(f"Query: {query}")
    print(f"Recursively retrieved {len(nodes)} nodes")
    for i, node in enumerate(nodes[:3], 1):
        print(f"{i}. Score: {node.score:.4f}" if hasattr(node, 'score') and node.score else f"{i}. (Recursive)")
        print(f"   Text: {node.text[:100]}...")
        print()
except Exception as e:
    print(f"Query: {query}")
    print(f"Recursive retriever demo: {str(e)}")
    print("Note: Recursive retriever requires specific node reference setup")
    
    # Fallback to basic retrieval for demonstration
    print("\nFalling back to basic retrieval demonstration...")
    base_nodes = base_retriever.retrieve(query)
    for i, node in enumerate(base_nodes[:2], 1):
        print(f"{i}. Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
```

## 6. Query Fusion Retriever - Multi-Query Enhancement with Advanced Fusion

The Query Fusion Retriever **combines results from different retrievers** (such as vector-based and keyword-based methods) and **optionally generates multiple variations of a query using an LLM to improve coverage**. **The results are merged using fusion strategies** to improve recall.

**How it works (from authoritative source)**:
- **Combines results from multiple retrievers** - e.g., vector-based and keyword-based methods
- **Supports multiple query variations** - generates different formulations of the same query
- **Uses fusion strategies to improve recall** - sophisticated merging techniques
- **Improved Coverage**: Reduces impact of query formulation on final results

**Core capabilities**:
1. **Multiple Retriever Support**: Combines results from different retrievers
2. **Query Variation Generation**: Optionally generates multiple variations of a query using an LLM
3. **Fusion Strategies**: Merges results using sophisticated fusion techniques

**Fusion Strategies Supported (from authoritative source)**:
1. **Reciprocal Rank Fusion (RRF)**: **Combines rankings across queries** - robust and doesn't rely on score magnitudes
2. **Relative Score Fusion**: **Normalizes scores within each result set** - preserves the relative confidence of each retriever
3. **Distribution-Based Fusion**: **Uses statistical normalization** - ideal for handling score variability

**When to use (based on authoritative guidance):**
- General Q&A where you want to combine semantic relevance with keyword matching
- Complex or ambiguous queries that may benefit from multiple formulations
- When query phrasing significantly impacts results
- Research and exploratory search scenarios
- When users provide under-specified or unclear queries

**Configuration:**
- `num_queries`: Number of query variations to generate (default: 4)
- `mode`: Fusion strategy ("reciprocal_rerank", "relative_score", "dist_based_score")
- `similarity_top_k`: Number of results to retrieve per query
- `use_async`: Enable async processing for better performance

**Key benefit**: **Uses fusion strategies such as reciprocal rank fusion or relative score fusion** to intelligently combine results

**Strengths**: 
- Improved recall through multiple query formulations
- Handles query variations effectively
- Reduces query sensitivity
- Combines strengths of different retrieval methods

**Limitations**: 
- Higher computational cost due to multiple retrievers/queries
- Requires LLM for query generation (additional cost)
- May introduce noise if fusion strategies are not well-tuned
- More complex setup and configuration


```python
print("=" * 60)
print("6. QUERY FUSION RETRIEVER - OVERVIEW")
print("=" * 60)

# Create base retriever
base_retriever = lab.vector_index.as_retriever(similarity_top_k=3)

query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"
print(f"Query: {query}")
print("QueryFusionRetriever generates multiple query variations and fuses results")
print("using one of three sophisticated fusion modes.")

print("\nOverview of Fusion Modes:")
print("1. RECIPROCAL_RERANK: Uses reciprocal rank fusion (most robust)")
print("2. RELATIVE_SCORE: Preserves score magnitudes (most interpretable)")  
print("3. DIST_BASED_SCORE: Statistical normalization (most sophisticated)")

print("\nDemonstration workflow:")
print("Each subsection below explores one fusion mode in detail with:")
print("- Theoretical explanation of the fusion method")
print("- Live demonstration using QueryFusionRetriever")
print("- Manual implementation showing the underlying mathematics")
print("- Use case recommendations and trade-offs")

print(f"\nUsing consistent test query throughout: '{query}'")
print("This allows direct comparison of how each fusion mode handles the same input.")

print("\nProceed to subsections 6.1, 6.2, and 6.3 for detailed demonstrations...")
```

### 6.1 Reciprocal Rank Fusion (RRF) Mode

Reciprocal Rank Fusion is the most robust fusion method in QueryFusionRetriever, designed to combine ranked lists from multiple query variations by using the reciprocal of ranks, which reduces the impact of outliers and provides stable fusion results.

**How it works within QueryFusionRetriever**:
- Generates multiple query variations (e.g., "machine learning approaches", "ML techniques", "learning algorithms")
- Retrieves results for each query variation
- Calculates reciprocal rank score: `1 / (rank + k)` where k is typically 60
- Sums reciprocal rank scores across all query variations for each document
- Re-ranks documents by combined RRF scores

**Mathematical formula**:
```
RRF_score(d) = Σ (1 / (rank_i(d) + k))
```
Where:
- `d` is a document
- `rank_i(d)` is the rank of document d in query variation i's results
- `k` is a constant (typically 60) that controls the fusion behavior

**Why RRF works well for query fusion**:
- **Scale-invariant**: Works regardless of individual query result score ranges
- **Robust to outliers**: Reciprocal function reduces impact of extreme rankings
- **Query-agnostic**: Doesn't depend on specific query formulations
- **Proven effectiveness**: Well-established in information retrieval research

**When to use RRF mode**:
- Default choice for most query fusion scenarios
- When query variations might have very different result qualities
- When you want stable, predictable fusion behavior
- For production systems requiring consistent performance

**Advantages**:
- Most stable fusion method across different query types
- No parameter tuning required beyond the standard k=60
- Handles varying numbers of results per query variation gracefully
- Computationally efficient

**Limitations**:
- Loses absolute score information from individual queries
- Treats all query variations equally (no weighting)
- May not leverage score magnitude differences effectively

*Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion/*


```python
print("=" * 60)
print("6.1 RECIPROCAL RANK FUSION MODE DEMONSTRATION")
print("=" * 60)

# Create QueryFusionRetriever with RRF mode
base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

print("Testing QueryFusionRetriever with reciprocal_rerank mode:")
print("This demonstrates how RRF works within the query fusion framework")

# Use the same query for consistency across all fusion modes
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with RRF mode
    rrf_query_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=True
    )
    
    print(f"\nQuery: {query}")
    print("QueryFusionRetriever will:")
    print("1. Generate query variations using LLM")
    print("2. Retrieve results for each variation")
    print("3. Apply Reciprocal Rank Fusion")
    
    nodes = rrf_query_fusion.retrieve(query)
    
    print(f"\nRRF Query Fusion Results:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Final RRF Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
    
    print("RRF Benefits in Query Fusion Context:")
    print("- Automatically handles query variations of different quality")
    print("- No bias toward queries that return higher raw scores")
    print("- Stable performance across diverse query formulations")
    
except Exception as e:
    print(f"QueryFusionRetriever error: {e}")
    print("Demonstrating RRF concept manually with query variations...")
    
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual RRF with Query Variations:")
    all_results = {}
    
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        
        # Apply RRF scoring
        for rank, node in enumerate(nodes):
            node_id = node.node.node_id
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'rrf_score': 0,
                    'query_ranks': []
                }
            
            # Calculate RRF contribution: 1 / (rank + k)
            k = 60  # Standard RRF parameter
            rrf_contribution = 1.0 / (rank + 1 + k)
            all_results[node_id]['rrf_score'] += rrf_contribution
            all_results[node_id]['query_ranks'].append((i, rank + 1))
    
    # Sort by final RRF score
    sorted_results = sorted(
        all_results.values(), 
        key=lambda x: x['rrf_score'], 
        reverse=True
    )
    
    print(f"\nCombined RRF Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Final RRF Score: {result['rrf_score']:.4f}")
        print(f"   Query ranks: {result['query_ranks']}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("RRF Formula Demonstration:")
    print("For each document: RRF_score = Σ(1 / (rank + 60))")
    print("- Rank 1 in query: 1/(1+60) = 0.0164")
    print("- Rank 2 in query: 1/(2+60) = 0.0161")
    print("- Rank 3 in query: 1/(3+60) = 0.0159")
    print("Documents appearing in multiple queries get higher combined scores")
```

### 6.2 Relative Score Fusion Mode

Relative Score Fusion normalizes retrieval scores relative to the maximum score within each query variation's results, enabling effective combination when you want to preserve score magnitude information across different query formulations.

**How it works within QueryFusionRetriever**:
- Generates multiple query variations using LLM
- Retrieves results for each query variation
- Normalizes each query's scores by dividing by the maximum score in that query's results
- Creates scores in the range [0, 1] where 1 is the best result from each query variation
- Combines normalized scores using weighted average or sum

**Mathematical approach**:
```
normalized_score_i(d) = score_i(d) / max_score_i
combined_score(d) = Σ (weight_i × normalized_score_i(d))
```

**Why Relative Score Fusion is valuable for query variations**:
- **Preserves score magnitudes**: Unlike RRF, retains information about how confident each query was about its results
- **Fair combination**: Ensures no single query variation dominates due to different scoring scales
- **Interpretable results**: Final scores reflect the relative strength across query variations
- **Flexible weighting**: Can weight certain query formulations more heavily if desired

**When to use Relative Score mode**:
- When you trust the embedding model's confidence scores
- For queries where score magnitudes are meaningful
- When different query variations should contribute proportionally to their confidence
- In scenarios where you want to understand why certain results ranked highly

**Configuration within QueryFusionRetriever**:
- Automatically handles score normalization across query variations
- Equal weighting of all query variations by default
- Preserves relative differences in retriever confidence

**Advantages**:
- Preserves valuable score magnitude information
- Intuitive normalization approach
- Works well when retriever scores are reliable
- More interpretable than pure rank-based methods

**Limitations**:
- Sensitive to outlier scores within individual query results
- Assumes retriever scores are meaningful and comparable
- May not handle unreliable scoring mechanisms well

*Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/relative_score_dist_fusion/*


```python
print("=" * 60)
print("6.2 RELATIVE SCORE FUSION MODE DEMONSTRATION")
print("=" * 60)

base_retriever = lab.vector_index.as_retriever(similarity_top_k=5)

print("Testing QueryFusionRetriever with relative_score mode:")
print("This mode preserves score magnitudes while normalizing across query variations")

# Use the same query for consistency across all fusion modes
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with relative score mode
    rel_score_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="relative_score",
        use_async=False,
        verbose=False
    )
    
    print(f"\nQuery: {query}")
    print("QueryFusionRetriever with relative_score will:")
    print("1. Generate query variations")
    print("2. Normalize scores within each variation (score/max_score)")
    print("3. Combine normalized scores")
    
    nodes = rel_score_fusion.retrieve(query)
    
    print(f"\nRelative Score Fusion Results:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Combined Relative Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
    
    print("Relative Score Benefits in Query Fusion:")
    print("- Preserves confidence information from embedding model")
    print("- Ensures fair contribution from each query variation")
    print("- More interpretable than rank-only methods")
    
except Exception as e:
    print(f"QueryFusionRetriever error: {e}")
    print("Demonstrating Relative Score concept manually...")
    
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual Relative Score Fusion with Query Variations:")
    all_results = {}
    query_max_scores = []
    
    # Step 1: Get results and find max scores for each query
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        max_score = max(scores) if scores else 1.0
        query_max_scores.append(max_score)
        
        print(f"Max score for this query: {max_score:.4f}")
        
        # Store results with normalization info
        for node in nodes:
            node_id = node.node.node_id
            original_score = node.score or 0
            normalized_score = original_score / max_score if max_score > 0 else 0
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': original_score,
                'normalized': normalized_score
            })
    
    # Step 2: Sort by combined relative score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    print(f"\nCombined Relative Score Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Combined Score: {result['combined_score']:.4f}")
        print(f"   Score breakdown:")
        for contrib in result['contributions']:
            print(f"     Query {contrib['query']}: {contrib['original']:.3f} → {contrib['normalized']:.3f}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("Relative Score Normalization Process:")
    print("1. For each query variation, find max_score")
    print("2. Normalize: normalized_score = original_score / max_score")
    print("3. Sum normalized scores across all query variations")
    print("4. Documents with consistently high scores across queries win")
```

### 6.3 Distribution-Based Score Fusion Mode

Distribution-Based Score Fusion uses statistical properties of score distributions from each query variation to normalize and combine retrieval results, providing the most sophisticated handling of score variability and reliability across different query formulations.

**How it works within QueryFusionRetriever**:
- Generates multiple query variations using LLM
- Analyzes the statistical distribution of scores from each query variation
- Normalizes scores using distribution parameters (mean, standard deviation, percentiles)
- Applies statistical transformations like z-score normalization or percentile ranking
- Combines normalized scores with confidence weighting based on distribution characteristics

**Statistical approaches used**:
1. **Z-score normalization**: Centers scores around mean with unit variance
   - Formula: `z_score = (score - mean) / std_dev`
   - Converts to [0,1] range using sigmoid: `1 / (1 + exp(-z_score))`

2. **Percentile ranking**: Converts scores to percentile positions
   - Formula: `percentile = rank(score) / total_results`

3. **Distribution-aware normalization**: Considers score distribution shape
   - Uses IQR (Interquartile Range) to adjust for distribution spread
   - Handles multi-modal distributions from different query variations

**Why Distribution-Based Fusion excels for query variations**:
- **Statistical robustness**: Accounts for how scores are distributed within each query variation
- **Adaptive weighting**: Can weight query variations based on their score distribution confidence
- **Outlier handling**: Statistical methods naturally handle extreme scores
- **Multi-modal support**: Each query variation may have different score distribution characteristics

**When to use Distribution-Based mode**:
- When query variations produce very different score distributions
- For complex queries where some variations are much more reliable than others
- When you need statistically principled score combination
- In scenarios with noisy or unreliable retrieval scoring

**Advanced features in QueryFusionRetriever context**:
- Automatic distribution analysis for each query variation
- Confidence-based weighting of query variations
- Robust handling of varying result set sizes
- Statistical outlier detection within query results

**Advantages**:
- Most statistically principled approach to query fusion
- Handles complex score distributions effectively
- Adapts to different query variation characteristics
- Robust to various types of score variability and noise

**Limitations**:
- Most computationally intensive fusion method
- Requires sufficient results for reliable distribution estimation
- May over-normalize in some simple scenarios
- More complex to interpret than simpler fusion methods

*Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/relative_score_dist_fusion/*


```python
print("=" * 60)
print("6.3 DISTRIBUTION-BASED SCORE FUSION MODE DEMONSTRATION")
print("=" * 60)

base_retriever = lab.vector_index.as_retriever(similarity_top_k=8)

print("Testing QueryFusionRetriever with dist_based_score mode:")
print("This mode uses statistical analysis for the most sophisticated score fusion")

# Use the same query for consistency across all fusion modes
query = DEMO_QUERIES["comprehensive"]  # "What are the main approaches to machine learning?"

try:
    # Create query fusion retriever with distribution-based mode
    dist_fusion = QueryFusionRetriever(
        [base_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="dist_based_score",
        use_async=False,
        verbose=False
    )
    
    print(f"\nQuery: {query}")
    print("QueryFusionRetriever with dist_based_score will:")
    print("1. Generate query variations")
    print("2. Analyze score distributions for each variation")
    print("3. Apply statistical normalization (z-score, percentiles)")
    print("4. Combine with distribution-aware weighting")
    
    nodes = dist_fusion.retrieve(query)
    
    print(f"\nDistribution-Based Fusion Results:")
    for i, node in enumerate(nodes, 1):
        print(f"{i}. Statistically Normalized Score: {node.score:.4f}")
        print(f"   Text: {node.text[:100]}...")
        print()
    
    print("Distribution-Based Benefits in Query Fusion:")
    print("- Accounts for score distribution differences between query variations")
    print("- Statistically robust against outliers and noise")
    print("- Adapts weighting based on query variation reliability")
    
except Exception as e:
    print(f"QueryFusionRetriever error: {e}")
    print("Demonstrating Distribution-Based concept manually...")
    
    if not SCIPY_AVAILABLE:
        print("⚠️ Full statistical analysis requires scipy")
    
    # Manual demonstration with query variations derived from the main query
    query_variations = [
        DEMO_QUERIES["comprehensive"],  # Original query
        "machine learning approaches and methods",
        "different ML techniques and algorithms"
    ]
    
    print("Manual Distribution-Based Fusion with Query Variations:")
    all_results = {}
    variation_stats = []
    
    # Step 1: Collect results and analyze distributions
    for i, query_var in enumerate(query_variations):
        print(f"\nQuery variation {i+1}: {query_var}")
        nodes = base_retriever.retrieve(query_var)
        scores = [node.score or 0 for node in nodes]
        
        # Calculate distribution statistics
        mean_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if len(scores) > 1 else 1
        min_score = np.min(scores) if scores else 0
        max_score = np.max(scores) if scores else 1
        
        stats_info = {
            'mean': mean_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'nodes': nodes,
            'scores': scores
        }
        variation_stats.append(stats_info)
        
        print(f"Distribution stats: mean={mean_score:.3f}, std={std_score:.3f}")
        print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")
        
        # Apply z-score normalization
        for node, score in zip(nodes, scores):
            node_id = node.node.node_id
            
            # Z-score normalization
            if std_score > 0:
                z_score = (score - mean_score) / std_score
            else:
                z_score = 0
            
            # Convert to [0,1] using sigmoid
            normalized_score = 1 / (1 + np.exp(-z_score))
            
            if node_id not in all_results:
                all_results[node_id] = {
                    'node': node,
                    'combined_score': 0,
                    'contributions': []
                }
            
            all_results[node_id]['combined_score'] += normalized_score
            all_results[node_id]['contributions'].append({
                'query': i,
                'original': score,
                'z_score': z_score,
                'normalized': normalized_score
            })
    
    # Step 2: Sort by combined distribution-based score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    print(f"\nCombined Distribution-Based Results (top 3):")
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Combined Score: {result['combined_score']:.4f}")
        print(f"   Statistical breakdown:")
        for contrib in result['contributions']:
            print(f"     Query {contrib['query']}: {contrib['original']:.3f} → "
                  f"z={contrib['z_score']:.2f} → {contrib['normalized']:.3f}")
        print(f"   Text: {result['node'].text[:100]}...")
        print()
    
    print("Distribution-Based Process:")
    print("1. Calculate mean and std for each query variation")
    print("2. Z-score normalize: z = (score - mean) / std")
    print("3. Sigmoid transform: normalized = 1 / (1 + exp(-z))")
    print("4. Sum normalized scores across variations")
    print("5. Results reflect statistical significance across all query forms")

# Show fusion mode comparison summary
print("\n" + "=" * 60)
print("FUSION MODES COMPARISON SUMMARY")
print("=" * 60)
print("All three modes tested with the same query for direct comparison:")
print(f"Query: {query}")
print()
print("Mode Characteristics:")
print("• RRF (reciprocal_rerank): Most robust, rank-based, scale-invariant")
print("• Relative Score: Preserves confidence, normalizes by max score")  
print("• Distribution-Based: Most sophisticated, statistical normalization")
print()
print("Choose based on your use case:")
print("- Production stability → RRF")
print("- Score interpretability → Relative Score")
print("- Statistical robustness → Distribution-Based")
```

## Recommended Retrievers by Use Case

Based on the authoritative source and the characteristics of each retriever, here are recommended approaches for different scenarios:

**General Q&A Applications:**
- **Primary**: Vector Index Retriever for semantic understanding
- **Enhancement**: Combine with BM25 Retriever using Query Fusion for hybrid approach
- **Benefit**: Combines semantic relevance with keyword matching
- **From authoritative source**: "For general Q&A, use a vector index retriever, potentially combined with a BM25 retriever. This retriever fusion combines semantic relevance with keyword matching."

**Technical Documentation:**
- **Primary**: BM25 Retriever for exact term matching
- **Enhancement**: Vector Index Retriever as secondary for contextual flexibility
- **Benefit**: Prioritizes exact technical terms while maintaining semantic understanding
- **From authoritative source**: "For technical documents, especially those where exact terms need to be prioritized, consider making BM25 your primary retriever, with the vector index retriever adding contextual flexibility as a secondary retriever."

**Long Documents:**
- **Primary**: Auto Merging Retriever
- **Benefit**: Retrieves longer parent versions only if enough shorter child versions are retrieved, preserving context
- **From authoritative source**: "For long documents, the auto merging retriever is a great option, because it will retrieve longer parent versions only if enough shorter child versions are retrieved."

**Research Papers:**
- **Primary**: Recursive Retriever
- **Benefit**: Follows citations and references to retrieve relevant content from cited papers
- **From authoritative source**: "For research papers, use the recursive retriever in order to retrieve relevant content from cited papers."

**Large Document Collections:**
- **Primary**: Document Summary Index Retriever for initial filtering
- **Enhancement**: Followed by Vector Index Retriever for detailed search within relevant documents
- **Benefit**: Narrows down relevant documents first, then performs detailed retrieval
- **From authoritative source**: "For large document sets, consider using the document summary index retriever to narrow down the number of relevant documents, followed by a vector search within the remaining subset to retrieve the most pertinent content."


---

# Exercises

Now that you've learned about advanced retrievers, let's practice implementing them in different scenarios.


## Exercise 1 - Build a Custom Hybrid Retriever

Your task is to create a hybrid retriever that combines both vector similarity and BM25 keyword search for improved results.

**Requirements:**
- Use both Vector Index Retriever and BM25 Retriever
- Implement a simple score fusion mechanism which takes a weighted average of normalized scores
- Test with different query types (semantic vs keyword-focused)

**Important Note**: Node IDs from different retrievers won't match even for the same content, so we need to match by text content instead.

```python
# TODO: Implement hybrid retriever
# Step 1: Create both retrievers
vector_retriever = # Your code here
bm25_retriever = # Your code here

# Step 2: Implement score fusion
def hybrid_retrieve(query, top_k=5):
    # Your implementation here
    pass

# Step 3: Test with different queries
test_queries = [
    "What is machine learning?",  # Semantic query
    "neural networks deep learning",  # Keyword query
    "supervised learning techniques"  # Mixed query
]
```


```python
### Put your solution here ###
```

<details>
    <summary>Click here for Solution</summary>

```python
# Create both retrievers
vector_retriever = lab.vector_index.as_retriever(similarity_top_k=10)
try:
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=lab.nodes, similarity_top_k=10
    )
except:
    # Fallback if BM25 is not available
    bm25_retriever = vector_retriever

def hybrid_retrieve(query, top_k=5):
    # Get results from both retrievers
    vector_results = vector_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)
    
    # Create dictionaries using text content as keys (since node IDs differ)
    vector_scores = {}
    bm25_scores = {}
    all_nodes = {}
    
    # Normalize vector scores
    max_vector_score = max([r.score for r in vector_results]) if vector_results else 1
    for result in vector_results:
        text_key = result.text.strip()  # Use text content as key
        normalized_score = result.score / max_vector_score
        vector_scores[text_key] = normalized_score
        all_nodes[text_key] = result
    
    # Normalize BM25 scores
    max_bm25_score = max([r.score for r in bm25_results]) if bm25_results else 1
    for result in bm25_results:
        text_key = result.text.strip()  # Use text content as key
        normalized_score = result.score / max_bm25_score
        bm25_scores[text_key] = normalized_score
        all_nodes[text_key] = result
    
    # Calculate hybrid scores
    hybrid_results = []
    for text_key in all_nodes:
        vector_score = vector_scores.get(text_key, 0)
        bm25_score = bm25_scores.get(text_key, 0)
        hybrid_score = 0.7 * vector_score + 0.3 * bm25_score
        
        hybrid_results.append({
            'node': all_nodes[text_key],
            'vector_score': vector_score,
            'bm25_score': bm25_score,
            'hybrid_score': hybrid_score
        })
    
    # Sort by hybrid score and return top k
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_results[:top_k]

# Test with different queries
test_queries = [
    "What is machine learning?",
    "neural networks deep learning", 
    "supervised learning techniques"
]

for query in test_queries:
    print(f"Query: {query}")
    results = hybrid_retrieve(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"   Vector: {result['vector_score']:.3f}, BM25: {result['bm25_score']:.3f}")
        print(f"   Text: {result['node'].text[:80]}...")
    print()
```

</details>


## Exercise 2 - Create a Production RAG Pipeline

Build a complete RAG pipeline that uses multiple retrieval strategies and includes evaluation metrics.

**Requirements:**
- Implement retrieval with multiple strategies
- Add query routing logic
- Include basic evaluation metrics that evaluate whether the pipeline succeeded or failed
- Handle edge cases and errors

```python
# TODO: Implement production RAG pipeline
class ProductionRAGPipeline:
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        # Your initialization code here
    
    def query(self, question, strategy="auto"):
        # Your implementation here
        pass
    
    def evaluate(self, test_queries, expected_answers):
        # Your evaluation implementation here
        pass

# Test the pipeline
pipeline = ProductionRAGPipeline(lab.vector_index, llm)
```


```python
### Put your solution here ###
```

<details>
    <summary>Click here for Solution</summary>

```python
class ProductionRAGPipeline:
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.vector_retriever = index.as_retriever(similarity_top_k=5)
        
    def _route_query(self, question):
        """Simple query routing based on question characteristics"""
        if any(word in question.lower() for word in ["what", "explain", "describe"]):
            return "semantic"
        elif any(word in question.lower() for word in ["list", "types", "examples"]):
            return "comprehensive"
        else:
            return "semantic"
    
    def query(self, question, strategy="auto"):
        try:
            # Route query if strategy is auto
            if strategy == "auto":
                strategy = self._route_query(question)
            
            # Retrieve relevant documents
            if strategy == "semantic":
                retriever = self.vector_retriever
                top_k = 3
            elif strategy == "comprehensive":
                retriever = self.vector_retriever
                top_k = 5
            else:
                retriever = self.vector_retriever
                top_k = 3
            
            # Get relevant documents
            relevant_docs = retriever.retrieve(question)
            
            # Prepare context
            context = "\n\n".join([doc.text for doc in relevant_docs[:top_k]])
            
            # Generate response
            prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {question}

Answer:"""
            
            try:
                response = self.llm.complete(prompt)
                return {
                    "answer": response.text,
                    "strategy": strategy,
                    "num_docs": len(relevant_docs),
                    "status": "success"
                }
            except Exception as e:
                return {
                    "answer": f"Based on the retrieved documents: {context[:200]}...",
                    "strategy": strategy,
                    "num_docs": len(relevant_docs),
                    "status": f"llm_error: {str(e)}"
                }
                
        except Exception as e:
            return {
                "answer": "I encountered an error processing your question.",
                "strategy": strategy,
                "num_docs": 0,
                "status": f"error: {str(e)}"
            }
    
    def evaluate(self, test_queries):
        results = []
        for query in test_queries:
            result = self.query(query)
            results.append({
                "query": query,
                "result": result,
                "success": result["status"] == "success"
            })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        return {
            "success_rate": success_rate,
            "results": results
        }

# Test the pipeline
pipeline = ProductionRAGPipeline(lab.vector_index, llm)

test_queries = [
    "What is machine learning?",
    "List different types of learning algorithms",
    "Explain neural networks"
]

print("Testing Production RAG Pipeline:")
for query in test_queries:
    result = pipeline.query(query)
    print(f"\nQuery: {query}")
    print(f"Strategy: {result['strategy']}")
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer'][:100]}...")

# Evaluate performance
evaluation = pipeline.evaluate(test_queries)
print(f"\nPipeline Success Rate: {evaluation['success_rate']:.2%}")
```

</details>


## Summary

Congratulations! You've successfully learned about advanced retrievers in LlamaIndex and implemented several practical examples. Here's what you've accomplished:

**Key Concepts Mastered:**
- **Vector Index Retriever**: Semantic search using embeddings
- **BM25 Retriever**: Advanced keyword-based search with TF-IDF improvements
- **Document Summary Index**: Intelligent document selection using summaries
- **Auto Merging Retriever**: Hierarchical context preservation
- **Recursive Retriever**: Multi-level reference following
- **Query Fusion Retriever**: Multi-query enhancement with three fusion modes

**Practical Skills Developed:**
- Implementing hybrid retrieval strategies
- Combining different retrieval methods effectively
- Building production-ready RAG pipelines
- Evaluating retrieval performance

**Best Practices Learned:**
- When to use each retrieval method
- How to combine multiple retrieval strategies
- Production considerations for RAG systems
- Evaluation techniques for retrieval quality

**Next Steps:**
- Experiment with different embedding models
- Implement more sophisticated fusion techniques
- Add reranking models for improved precision
- Scale to larger document collections
- Integrate with production systems

---


## Authors

[Wojciech \"Victor\" Fulmyk](https://www.linkedin.com/in/wfulmyk)

Wojciech "Victor" Fulmyk is a Data Scientist at IBM

<!--## Change Log

<details>
    <summary>Click here for the changelog</summary>

|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2025-07-18|0.1|Wojciech "Victor" Fulmyk|Initial version|

</details>
-->

---

Copyright © IBM Corporation. All rights reserved.


