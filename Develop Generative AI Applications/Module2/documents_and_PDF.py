

import os
import sys
import time

from langchain_core.prompts import ChatPromptTemplate
# Import the Document class from langchain_core.documents module
# Document is a container for text content with associated metadata
from langchain_core.documents import Document
# Import the PyPDFLoader class from langchain_community's document_loaders module
# This loader is specifically designed to load and parse PDF files
from langchain_community.document_loaders import PyPDFLoader





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

# Create a Document instance with:
# 1. page_content: The actual text content about Python
# 2. metadata: A dictionary containing additional information about this document
Document(page_content="""Python is an interpreted high-level general-purpose programming language.
 Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
metadata={
    'my_document_id' : 234234,                      # Unique identifier for this document
    'my_document_source' : "About Python",          # Source or title information
    'my_document_create_time' : int(time.time())          # Unix timestamp for document creation (March 28, 2023)
 })
if __name__ == "__main__":
    # Create a PyPDFLoader instance by passing the URL of the PDF file
    # The loader will download the PDF from the specified URL and prepare it for loading
    loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")

    print("Loading PDF from URL...")
    document = loader.load()

    print("\n--- Page 2 Content (First 1000 characters) ---")
    # Note: document[1] is actually the 2nd page since Python is 0-indexed!
    print(document[1].page_content[:1000])
