# Import core to trigger .env loading and warning suppressions!
import core

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

from langchain_community.vectorstores import Chroma

if __name__ == "__main__":
    # 1. Scrape a soccer webpage
    print("Scraping soccer webpage...")
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Association_football")
    web_data = loader.load()
    # ✨ Data Cleaning Step using Regex!
    import re
    print("Cleaning data with Regex...")
    for document in web_data:
        # re.sub() finds a pattern and replaces it with an empty string ''
        # Pattern: \[[0-9]+\] means "Find a bracket, then numbers, then a closing bracket"
        document.page_content = re.sub(r'\[[0-9]+\]', '', document.page_content)

    # 2. Split into chunks using the smarter Recursive splitter!
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(web_data)
    print(f"Created {len(chunks)} chunks!")

    # 3. Configure embedding parameters
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    # 4. Create WatsonxEmbeddings instance
    print("Connecting to Watsonx...")
    import os
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=embed_params,
    )

    docsearch = Chroma.from_documents(
        chunks,
        watsonx_embedding
        )
    # Use the docsearch vector store as a retriever
    # This converts the vector store into a retriever interface that can fetch relevant documents

    # Example 1: Default Retriever (Returns top 4 results)
    retriever1 = docsearch.as_retriever()
    docs1 = retriever1.invoke("fifa")
    print(f"\n📚 Default Librarian brought back {len(docs1)} documents!")

    # Example 2: Bring exactly 2 results (k=2)
    retriever2 = docsearch.as_retriever(search_kwargs={"k": 2})
    docs2 = retriever2.invoke("fifa")
    print(f"📚 Filtered Librarian (k=2) brought back {len(docs2)} documents!")

    # Example 3: Only bring results if they are an 80% match or higher
    retriever3 = docsearch.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.8}
    )
    docs3 = retriever3.invoke("fifa")
    print(f"📚 Threshold Librarian brought back {len(docs3)} documents!")

    # Example 4: Randomize the results slightly so the Chatbot doesn't get bored (MMR)
    retriever4 = docsearch.as_retriever(search_type="mmr")
    docs4 = retriever4.invoke("fifa")
    print(f"📚 MMR Librarian brought back {len(docs4)} documents!")

    print("\n" + "="*50)
    print("🔍 MMR RETRIEVER RESULTS:")
    print("="*50)
    for i, doc in enumerate(docs4, 1):
        print(f"\n📄 Document {i}:")
        print("-" * 50)
        # .strip() removes any awkward empty lines at the start/end of the chunk
        print(doc.page_content.strip())
    print("\n" + "="*50)