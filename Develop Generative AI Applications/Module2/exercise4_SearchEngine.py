import os
from dotenv import load_dotenv;

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.vectorstores import Chroma

# Additional imports for Parent Document Retriever
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

load_dotenv()


# 1. Load a document about AI
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
web_data = loader.load()

# 2. Split the document into chunks
print("Configuring splitters...")
# The parent chunks are huge (2000 chars) to give the LLM full context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) 
# The child chunks are tiny (400 chars) to make the math search extremely accurate
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 3. Configure embedding parameters
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
# 4. Setup Storage
# The filing cabinet for the tiny math chunks

vectorstore= Chroma(
    collection_name="qa_split_parents",
    embedding_function=watsonx_embedding
)

# 5.The storage room for the massive Parent text blocks
store = InMemoryStore()

# 6. Create the Smarter Librarian (ParentDocumentRetriever)
retriever= ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4, "k": 3}
)

# 7. Ingest Data
print("Chopping documents and calculating vectors (This may take ~15 seconds)...")
retriever.add_documents(web_data)

# 8. Define a function to search for relevant information
def search_documents(query:str, top_k:int=3)->list[Document]:
    docs =retriever.invoke(query)

    return docs[:top_k]


# 9. Test with a few queries
test_queries = [
    "What is LangChain?",
    "How do retrievers work?",
    "Why is document splitting important?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    docs:list[Document] = search_documents(query, 3)
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Document {i}:")
        print("-" * 50)
        # .strip() removes any awkward empty lines at the start/end of the chunk
        print(doc.page_content.strip())
    print("\n" + "="*50)