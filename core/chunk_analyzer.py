from typing import List, Any

def display_document_stats(docs: List[Any], name: str) -> None:
    """Display detailed statistics about a list of LangChain document chunks."""
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
