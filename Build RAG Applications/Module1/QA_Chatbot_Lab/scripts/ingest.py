

import os
import wget
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def setup_database():
    # Use absolute or relative paths that point to the new data folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, '../data/raw/companyPolicies.txt')
    url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

     # 1. Download the document if it doesn't exist
    if not os.path.exists(filename):
        print('Downloading document...')
        wget.download(url, out=filename)
        print("Downloada Complete...")
    # 2. Load the document
    loader= TextLoader(filename)
    documents= loader.load()

    # 3. Split the document into chunks
    text_splitter= CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts= text_splitter.split_documents(documents)
    print(f"Document split into {len(texts)} chunks.")

    # 4. Embed and store in ChromaDB locally
    embeddings= HuggingFaceEmbeddings()
    # We specify a persist_directory so the database is saved to disk
    persist_dir = os.path.join(base_dir, '../data/chroma_db')
    db= Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    print('Database successfully ingested and saved locally!')
if __name__ =="__main__":
    setup_database()