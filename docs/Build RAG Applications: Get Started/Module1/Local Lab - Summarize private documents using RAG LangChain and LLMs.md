# **Local Development: Summarize Private Documents Using RAG, LangChain, and LLMs**

##### Estimated time needed: **45** minutes

Imagine it's your first day at an exciting new job at a fast-growing tech company, Innovatech. Your manager, Alex, sends you a folder with private company policies and guidelines. You need to absorb this information quickly, but you cannot upload these private documents to public AI tools. 

Your colleague suggests creating a personal assistant using **Retrieval-Augmented Generation (RAG)** and **LangChain**. 

In this local development lab, we will translate the standard Jupyter Notebook lab into a robust, local Python application. You will learn how to set up your environment, ingest documents, and build a conversational AI agent that runs directly on your machine.

---

## __Table of Contents__

1. [Background](#background)
2. [Local Project Setup](#local-project-setup)
3. [Environment Variables](#environment-variables)
4. [Step 1: Ingesting the Document](#step-1-ingesting-the-document)
5. [Step 2: Building the RAG Agent](#step-2-building-the-rag-agent)
6. [Step 3: Adding Conversational Memory](#step-3-adding-conversational-memory)
7. [Running the Application](#running-the-application)
8. [Exercises](#exercises)

---

## Background

### What is RAG?
Retrieval-Augmented Generation (RAG) is a technique for augmenting LLM knowledge with additional data (like your private company documents). Since LLMs are trained on public data up to a certain point in time, RAG allows them to answer questions about private or new information by retrieving relevant document chunks and feeding them into the prompt.

### RAG Architecture
1. **Indexing**: Load the data, split it into smaller chunks, embed the text into vectors, and store it in a Vector Database (like Chroma).
2. **Retrieval and Generation**: Retrieve relevant chunks based on the user's question, and pass them to the LLM to generate a factual answer.

---

## Local Project Setup

Instead of running code in browser cells, we will create a proper Python project. 

1. **Create a new directory** for this project (e.g., `rag_summary_app`) and navigate into it.
2. **Create a virtual environment using `uv`**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install the required libraries**:
   Create a `requirements.txt` file:
   ```text
   ibm-watsonx-ai==0.2.6
   langchain==0.1.16
   langchain-ibm==0.1.4
   transformers==4.41.2
   huggingface-hub==0.23.4
   sentence-transformers==2.5.1
   chromadb
   wget==3.2
   python-dotenv
   ```
   Install them by running:
   ```bash
   uv pip install -r requirements.txt
   ```

---

## Environment Variables

In a local environment, we never hardcode API keys. Create a `.env` file in your project root:

```env
# .env
WATSONX_API_KEY="your_ibm_cloud_api_key"
WATSONX_PROJECT_ID="your_project_id"
WATSONX_URL="https://us-south.ml.cloud.ibm.com"
```

> **Note:** We will use `python-dotenv` to securely load these credentials into our application.

---

## Step 1: Ingesting the Document

Create a file named `ingest.py`. This script will download the document, split it into chunks, and store it in a local Chroma vector database.

```python
# ingest.py
import os
import wget
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def setup_database():
    filename = 'companyPolicies.txt'
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

    # 1. Download the document if it doesn't exist
    if not os.path.exists(filename):
        print("Downloading document...")
        wget.download(url, out=filename)
        print("\nDownload complete.")

    # 2. Load the document
    loader = TextLoader(filename)
    documents = loader.load()

    # 3. Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Document split into {len(texts)} chunks.")

    # 4. Embed and store in ChromaDB locally
    embeddings = HuggingFaceEmbeddings()
    # We specify a persist_directory so the database is saved to disk
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    print('Database successfully ingested and saved locally!')

if __name__ == "__main__":
    setup_database()
```

Run this script once to create your database:
```bash
python ingest.py
```

---

## Step 2 & 3: Building the RAG Agent with Memory

Now, create `app.py`. This script will load the vector database we just created, initialize the IBM Watsonx LLM, and start an interactive chat loop.

```python
# app.py
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# Load environment variables
load_dotenv()

def get_llm():
    model_id = 'ibm/granite-3-8b-instruct'
    
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
        GenParams.MAX_NEW_TOKENS: 256, 
        GenParams.TEMPERATURE: 0.5 
    }
    
    credentials = {
        "url": os.getenv("WATSONX_URL"),
        "apikey": os.getenv("WATSONX_API_KEY")
    }
    
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    return WatsonxLLM(model=model)

def main():
    print("Loading database...")
    embeddings = HuggingFaceEmbeddings()
    # Load the local Chroma database
    docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    llm = get_llm()
    
    # Initialize memory so the agent remembers previous questions
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)
    
    # Create the retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        memory=memory, 
        get_chat_history=lambda h: h, 
        return_source_documents=False
    )
    
    history = []
    
    print("\n========================================================")
    print("Welcome to your RAG Assistant! Ask me about company policies.")
    print("Type 'quit', 'exit', or 'bye' to stop.")
    print("========================================================\n")
    
    while True:
        query = input("\nQuestion: ")
        
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
            
        result = qa_chain.invoke({"question": query, "chat_history": history})
        
        # Save interaction to history
        history.append((query, result["answer"]))
        
        print("Answer:", result["answer"])

if __name__ == "__main__":
    main()
```

---

## Running the Application

To chat with your private documents locally, simply run:
```bash
python app.py
```

Try asking questions like:
- *"What is the mobile policy?"*
- *"Can I eat in company vehicles?"*
- *"What about smoking in them?"* (Notice how it remembers what "them" refers to!)

---

## Exercises

### Exercise 1: Work on your own document
Modify `ingest.py` to download and ingest a different document. You can try parsing the State of the Union address by changing the URL in `ingest.py` to:
`https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XVnuuEg94sAE4S_xAsGxBA.txt`

### Exercise 2: Return the source from the document
Sometimes you want to see exactly which part of the document the AI used to generate its answer. 
1. In `app.py`, change `return_source_documents=False` to `return_source_documents=True`.
2. Update the `print` statement to also output `result['source_documents'][0].page_content`.

### Exercise 3: Use another LLM model
Try swapping the IBM Granite model for an open-source Mistral model. In `app.py`, change the `model_id` to:
`model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'`
Restart the app and see how the responses differ!
