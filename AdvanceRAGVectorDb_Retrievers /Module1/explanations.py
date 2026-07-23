BACKGROUND_HELP = """
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

Sample Data Setup

We'll use a collection of AI and machine learning documents to demonstrate different retrieval strategies.
"""

VECTOR_INDEX_HELP = """
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

BM25_HELP = """
2. BM25 Retriever - Advanced Keyword-Based Search

BM25 is a keyword-based retrieval method that improves on TF-IDF by addressing some of its key limitations. It's widely used in production search systems including Elasticsearch and Apache Lucene.

### Understanding TF-IDF: The Foundation

Before diving into BM25, let's understand TF-IDF (Term Frequency-Inverse Document Frequency), which BM25 builds upon:

Term Frequency (TF): Measures how often a word appears in a document
- Example: If "neural" appears 3 times in a 100-word document, TF = 3/100 = 0.03

Inverse Document Frequency (IDF): Measures how rare a word is across all documents
- Example: If "neural" appears in only 2 out of 1000 documents, IDF = log(1000/2) = 6.21
- Common words like "the" have low IDF; rare technical terms have high IDF

TF-IDF Score: TF * IDF
- Highlights words that are frequent in one document but rare across the collection
- Developed by Karen Spärck Jones, who pioneered the concept of term specificity

### How BM25 Improves Upon TF-IDF

Key BM25 Improvements:

1. Term Frequency Saturation: BM25 reduces the impact of repeated terms using term frequency saturation
   - Problem: In TF-IDF, if a word appears 100 times vs 10 times, the score increases linearly
   - Solution: BM25 uses a saturation function that plateaus after a certain frequency

2. Document Length Normalization: BM25 adjusts for document length, making it more effective for keyword-based search
   - Problem: In TF-IDF, longer documents have unfair advantages
   - Solution: BM25 normalizes scores based on document length relative to average

3. Tunable Parameters: Allows fine-tuning for different types of content
   - k1 ≈ 1.2: Controls term frequency saturation (how quickly scores plateau)
   - b ≈ 0.75: Controls document length normalization (0=none, 1=full)

### When to Use BM25

Ideal for:
- Technical documentation where exact terms matter
- Legal documents with specific terminology
- Product catalogs with precise specifications
- Academic papers with specialized vocabulary
- Applications requiring keyword-based retrieval rather than semantic similarity

Advantages:
- Excellent precision for exact term matches
- Fast computational performance
- Proven effectiveness in production systems
- No training required (unlike neural approaches)
- Interpretable scoring mechanism

Limitations:
- No semantic understanding (doesn't handle synonyms)
- Struggles with typos and variations
- Limited context understanding
- Requires careful parameter tuning for optimal performance
"""
DOCUMENT_SUMMARY_HELP="""
Document Summary Index Retrievers

Document Summary Index Retrievers use document summaries instead of the actual documents to find relevant content, making them efficient for large collections. They return the original documents, not their summaries.

How it works (from authoritative source):

    Generates and stores summaries of documents at indexing time
    Uses summaries to filter documents before retrieving full content
    Two-stage Process: First uses summaries to filter documents, then returns full document content
    Especially useful for large, diverse corpora that cannot fit in the context window of an LLM

Two Retrieval Options:

    DocumentSummaryIndexLLMRetriever:
        Uses a large language model to analyze the query against document summaries
        Provides intelligent document selection but can be more time-consuming and expensive
        Best for complex queries requiring nuanced understanding

    DocumentSummaryIndexEmbeddingRetriever:
        Uses semantic similarity between the query and summary embeddings
        Faster and more cost-effective than LLM-based approach
        Good for straightforward similarity matching

When to use (based on authoritative guidance):

    Large document collections where documents cover different topics
    When you need efficient document-level filtering before detailed retrieval
    Multi-document QA where documents have distinct subject matters
    Large and diverse document sets that cannot fit in the context window of an LLM

Configuration Parameters:

    choice_top_k (LLM retriever): Number of documents to select
    similarity_top_k (Embedding retriever): Number of documents to select
    Default is 1, increase for multiple document retrieval

Key Point: Returns original documents, not their summaries - the summaries are only used for filtering

Strengths:

    Efficient document selection and reduces search space
    Good for heterogeneous collections with diverse topics
    Returns original documents with full context intact

Limitations:

    Requires LLM for summary generation during indexing
    May lose some detail present in original documents during summary creation
    LLM-based version can be slower and more expensive than other options

"""

AUTO_MERGING_RETRIEVER_HELP="""
Auto Merging Retriever - Hierarchical Context Preservation¶

Auto Merging Retriever is designed to preserve context in long documents using a hierarchical structure. It uses hierarchical chunking to break documents into parent and child nodes, and if enough child nodes from the same parent are retrieved, the retriever returns the parent node instead.

How it works (from authoritative source):

    Uses hierarchical chunking to break documents into parent and child nodes
    Retrieves parent if enough children match - intelligent merging logic
    Preserves context in long documents by consolidating related content
    Dual Storage: Smaller child chunks are indexed in the vector store for precise matching, while larger parent chunks are stored in the docstore

Key behavior pattern:

    Child chunks enable precise matching for specific queries
    When multiple child chunks from the same parent are retrieved, the system returns the parent chunk
    This helps consolidate related content and preserve broader context

When to use (based on authoritative guidance):

    Long documents where small chunks lose important surrounding context
    Legal documents, research papers, technical specifications that need context preservation
    When you need both precise matching and comprehensive context
    Documents with natural hierarchical structure (sections, subsections)

Configuration:

    chunk_sizes: List of chunk sizes from largest to smallest (e.g., [512, 256, 128])
    chunk_overlap: Overlap between chunks to maintain continuity
    Storage context manages both vector store (child nodes) and docstore (parent nodes)

Strengths:

    Automatically preserves context without manual intervention
    Reduces information fragmentation in long documents
    Intelligent merging based on retrieval patterns
    Maintains granular search capability while providing broader context

Limitations:

    More complex setup compared to basic retrievers
    Requires hierarchical document structure to be effective
    Higher storage overhead due to multiple chunk levels
    May not be suitable for very short documents

Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/

"""
RECURSIVE_RETRIEVER_HELP="""
Recursive Retriever - Multi-Level Reference Following

The Recursive Retriever is designed to follow relationships between nodes using references. It can follow references from one node to another, such as citations in academic papers or other metadata links, allowing it to retrieve related content across documents or layers of abstraction.

How it works (from authoritative source):

    Follows node references - traverses relationships to find referenced content
    Supports chunk and metadata linking - handles different types of references
    Multi-Level Navigation: Can execute sub-queries on referenced retrievers or query engines
    Network Building: Creates a network of interconnected retrievers that can reference each other

Reference Types Supported:

    Chunk References: Smaller child chunks refer to larger parent chunks for additional context
    Metadata References: Summaries or generated questions refer to larger content chunks, such as citations in academic papers

When to use (based on authoritative guidance):

    Academic papers with citations and extensive references
    Research papers where you need to retrieve relevant content from cited papers
    Documentation with cross-references and linked content
    Knowledge bases with interconnected information
    When nodes reference structured data (tables, databases, other documents)

Configuration:

    retriever_dict: Maps node IDs or keys to specific retrievers
    query_engine_dict: Maps keys to query engines for sub-queries
    Node metadata can contain references to other nodes or data structures

Key capability: Retrieves related content across documents by following reference chains

Strengths:

    Follows complex relationships and enables multi-step reasoning
    Provides comprehensive coverage across related documents
    Excellent for handling interconnected information systems
    Can traverse multiple levels of references automatically

Limitations:

    Requires careful setup of node relationships
    Can be computationally expensive for deep reference chains
    Complex debugging when reference chains are extensive
    May retrieve too much related content if not properly configured

Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/recurisve_retriever_nodes_braintrust/
"""