import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
import sys
# Ensure the root directory is in the Python path so we can import 'core'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

#Import the necessary document loaders to work with both PDF and web content.
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

#Load the provided paper about LangChain architecture.
loaderPDF = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf")


#Load contenty from Langchain Website.
web_url = "https://python.langchain.com/v0.2/docs/introduction/"
# Create a WebBaseLoader instance
loaderWEB = WebBaseLoader(web_url)

#Create two different text splitters with varying parameters.
# Create a CharacterTextSplitter with specific configuration:
# - chunk_size=200: Each chunk will contain approximately 200 characters
# - chunk_overlap=20: Consecutive chunks will overlap by 20 characters to maintain context
# - separator="\n": Text will be split at newline characters when possible
text_splitter1 = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
text_splitter2 = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator="\n")

def display_document_stats(docs, name):
    """Display statistics about a list of document chunks"""
    total_chunks = len(docs)
    total_chars = sum(len(doc.page_content) for doc in docs)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
    
    # Count unique metadata keys across all documents
    all_metadata_keys = set()
    for doc in docs:
        all_metadata_keys.update(doc.metadata.keys())
    
    # Print the statistics
    print(f"\n=== {name} Statistics ===")
    print(f"Total number of chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size:.2f} characters")
    print(f"Metadata keys preserved: {', '.join(all_metadata_keys)}")
    
    if docs:
        print("\nExample chunk:")
        example_doc = docs[min(5, total_chunks-1)]  # Get the 5th chunk or the last one if fewer
        print(f"Content (first 150 chars): {example_doc.page_content[:150]}...")
        print(f"Metadata: {example_doc.metadata}")
        
        # Calculate length distribution
        lengths = [len(doc.page_content) for doc in docs]
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"Min chunk size: {min_len} characters")
        print(f"Max chunk size: {max_len} characters")

if __name__ == "__main__":
    #Compare the resulting chunks from different splitters.
    print("Loading PDF from URL...")
    rawPDF = loaderPDF.load()
    print("Loading Content from URL...")
    rawWEB = loaderWEB.load()
    
    # Use splitter 1 for the PDF
    chunksPDF = text_splitter1.split_documents(rawPDF)
    
    # Use splitter 2 for the WEB 
    chunksWEB = text_splitter2.split_documents(rawWEB)

    display_document_stats(chunksPDF, "PDF with Splitter 1 (200 chars)")
    display_document_stats(chunksWEB, "WEB with Splitter 2 (400 chars)")