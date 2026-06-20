
import os
import sys

from langchain_community.document_loaders import WebBaseLoader
# Text splitters are used to divide large texts into smaller, manageable chunks
from langchain_text_splitters import CharacterTextSplitter



# Create a WebBaseLoader instance
loader = WebBaseLoader("https://webcatolicodejavier.org/")

# Create a CharacterTextSplitter with specific configuration:
# - chunk_size=200: Each chunk will contain approximately 200 characters
# - chunk_overlap=20: Consecutive chunks will overlap by 20 characters to maintain context
# - separator="\n": Text will be split at newline characters when possible
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")


if __name__ == "__main__":
    
    # 1. Load the web data
    print("Scraping website...")
    web_data = loader.load()
    
    # Split the previously loaded document (PDF or other text) into chunks
    # The split_documents method:
    # 1. Takes a list of Document objects
    # 2. Splits each document's content based on the configured parameters
    # 3. Returns a new list of Document objects where each contains a chunk of text
    # 4. Preserves the original metadata for each chunk
    chunks = text_splitter.split_documents(web_data)

    print(f"Total chunks created: {len(chunks)}")

    # Show what the resulting chunk objects look like!
    print("\n--- Inspecting the Document Object Structure ---")
    first_chunk = chunks[0]
    
    print(f"1. Object Type: {type(first_chunk)}")
    print(f"2. Metadata Dictionary: {first_chunk.metadata}")
    print(f"3. Text Content: \n{first_chunk.page_content}")
