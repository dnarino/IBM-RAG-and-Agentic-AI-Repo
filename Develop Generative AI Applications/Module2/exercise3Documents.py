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

from core.chunk_analyzer import display_document_stats

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