import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

if __name__ == "__main__":
    # 1. Scrape Wikipedia
    print("Scraping soccer webpage...")
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Association_football")
    web_data = loader.load()
    
    # Clean Data with Regex
    print("Cleaning data with Regex...")
    for document in web_data:
        document.page_content = re.sub(r'\[[0-9]+\]', '', document.page_content)

    # 2. Configure Splitters (The secret to ParentDocumentRetriever)
    # The parent chunks are huge (2000 chars) to give the Chatbot full context
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # The child chunks are tiny (400 chars) to make math search extremely accurate
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 3. Setup IBM Watsonx Embeddings
    print("Connecting to Watsonx...")
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=embed_params,
    )

    # 4. Setup Storage
    # The filing cabinet for the tiny math chunks
    vectorstore = Chroma(
        collection_name="split_parents", 
        embedding_function=watsonx_embedding
    )
    # The storage room for the massive Parent text blocks
    store = InMemoryStore()

    # 5. Create the Smarter Librarian
    retriever = ParentDocumentRetriever(
        # The vector store where child document embeddings will be stored and searched
        # This Chroma instance will contain the embeddings for the smaller chunks
   
        vectorstore=vectorstore,
        # The document store where parent documents will be stored
        # These larger chunks won't be embedded but will be retrieved by ID when needed
    
        docstore=store,
        # The splitter used to create small chunks (400 chars) for precise vector search
        # These smaller chunks are embedded and used for similarity matching
    
        child_splitter=child_splitter,
        # The splitter used to create larger chunks (2000 chars) for better context
        # These parent chunks provide more complete information when retrieved
    
        parent_splitter=parent_splitter,
    )

    # 6. Ingest Data
    print("Chopping documents and calculating vectors (This may take a moment)...")
    retriever.add_documents(web_data)
    
    # 7. Search!
    print("Searching for 'What happens if a game ends in a draw?'...")
    docs = retriever.invoke("What happens if a game ends in a draw?")

    print("\n" + "="*50)
    print("🔍 PARENT DOCUMENT RETRIEVER RESULTS:")
    print("="*50)
    print(f"Total Documents Found: {len(docs)}")
    
    for i, doc in enumerate(docs[:1], 1):  # Just print the top result so it doesn't flood terminal
        print(f"\n📄 Document {i}:")
        print(f"Total Character Length: {len(doc.page_content)} characters (This proves it's a massive Parent block!)")
        print("-" * 50)
        print(doc.page_content.strip())
    print("\n" + "="*50)

"""
The Senior Engineer's Rule of Thumb:

Use Standard Retrievers (Small Chunks): For FAQs, dictionaries, exact dates, and simple facts.
Use Parent Document Retrievers: For complex narratives, legal contracts, medical reports, 
or conversational codebases where missing the surrounding context would completely ruin the answer!
"""