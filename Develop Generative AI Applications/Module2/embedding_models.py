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

    # 5. Extract text from the chunks (we'll just embed the first 5 to save time/money)
    texts = [text.page_content for text in chunks[:5]]

    # 6. Generate the embeddings!
    print("Generating embeddings...")
    embedding_result = watsonx_embedding.embed_documents(texts)
    
    # 7. Inspect the result
    print("\n✅ Embedding Complete!")
    print("\n--- Inspecting the Vector ---")
    print(f"Original Text: {texts[0][:50]}...")
    print(f"Total Vector Dimensions: {len(embedding_result[0])}")
    print(f"First 5 Float Values of the Array: \n{embedding_result[0][:5]}")

    docsearch = Chroma.from_documents(chunks, watsonx_embedding)

    query="Soccer"
    docs = docsearch.similarity_search(query)
    print(docs[0].page_content)