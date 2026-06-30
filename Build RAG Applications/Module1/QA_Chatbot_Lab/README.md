# QA Chatbot Lab

This lab demonstrates how to use LangChain, ChromaDB, and IBM Watsonx to build a Retrieval-Augmented Generation (RAG) conversational chatbot. 

## Project Structure

Following production best practices, this project is structured as follows:

- **`data/raw/`**: Contains the raw text documents (e.g., `companyPolicies.txt`) to be analyzed.
- **`data/chroma_db/`**: The persistent local vector database. 
- **`scripts/`**: Contains one-off scripts, like `ingest.py`.
- **`src/`**: Contains the core application code (`main.py`, `models.py`).

*(Note: The `data/` folder is intentionally ignored by Git to prevent uploading private documents and massive databases to source control).*

## How to Run

### Step 1: Ingest Data
Before running the chatbot, you must first ingest the raw text documents, chunk them, embed them, and store them in the Chroma database. 

Run the ingestion script from the root of your project:
```bash
uv run python "Build RAG Applications/Module1/QA_Chatbot_Lab/scripts/ingest.py"
```

### Step 2: Run Chatbot
*(Code coming soon to `src/main.py`...)*
