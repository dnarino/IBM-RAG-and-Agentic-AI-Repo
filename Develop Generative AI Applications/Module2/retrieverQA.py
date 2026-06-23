import os
import re

from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.vectorstores import Chroma

# Additional imports for Parent Document Retriever
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

# Import our modularized client!
from core.watson_client import get_watsonx_chat

# 1. Define model and parameters
model_id='meta-llama/llama-3-3-70b-instruct'
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_k": 2,         # Strictly limit to the top 2 most likely words
    "top_p": 0.1        # Extremely low probability variance (highly factual)
}

# 2. Get the modularized LLM
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

# 3. Scrape a soccer webpage
print("Scraping soccer webpage...")
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Association_football")
web_data = loader.load()

# 4. Data Cleaning Step using Regex!
print("Cleaning data with Regex...")
for document in web_data:
    document.page_content = re.sub(r'\[[0-9]+\]', '', document.page_content)

# 5. Configure Splitters for Parent Document Strategy
print("Configuring splitters...")
# The parent chunks are huge (2000 chars) to give the LLM full context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# The child chunks are tiny (400 chars) to make the math search extremely accurate
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 6. Configure embedding parameters
print("Connecting to Watsonx Embeddings...")
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

# 7. Setup Storage
# The filing cabinet for the tiny math chunks
vectorstore = Chroma(
    collection_name="qa_split_parents", 
    embedding_function=watsonx_embedding
)
# The storage room for the massive Parent text blocks
store = InMemoryStore()

# 8. Create the Smarter Librarian (ParentDocumentRetriever)
retriever1 = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 9. Ingest Data
print("Chopping documents and calculating vectors (This may take ~15 seconds)...")
retriever1.add_documents(web_data)

# 10. Build the RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llama_LLM,
    chain_type="stuff",
    retriever=retriever1,  # Pass our massive Parent Document Retriever!
    return_source_documents=False
)

# 11. Run the Query!
query = "Explain the rules around what happensif a player start to sing in middle of a game."

print(f"\nAsking: '{query}'...")
response = qa.invoke(query)

print("\n" + "="*50)
print("🤖 AI Answer (Powered by Parent Document Context!):")
print("="*50)
print(response["result"])
print("="*50 + "\n")