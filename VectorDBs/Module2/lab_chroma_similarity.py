
from certifi import where
import ssl
# Bypass corporate SSL certificate verification issues on Hugging Face downloads
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
from chromadb.utils import embedding_functions


ef= embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

client = chromadb.HttpClient(host="localhost", port=8000)

collection_name="my_grocery_collection"

def is_query_list(query) -> list:
    if not isinstance(query, list):
        return [query]
    return query

def perform_similarity_search(collection, query_term: list) -> None:
    try:
        filtered_results = collection.query(
            query_texts=query_term,
            n_results=10
        )
        if not filtered_results or not filtered_results['ids'] or (len(filtered_results['documents']) == 0):
            print(f'There is no result with the query search {query_term}')
        print(f"Number of Results: {len(filtered_results['documents'][0])}")
        print(f'Top 4 similar documents to "{query_term[0]}":')
        # Access the nested arrays in 'filtered_results["ids"]' and 'results["filtered_results"]'
        for i in range(min(4, len(filtered_results['ids'][0]))):
            doc_id = filtered_results['ids'][0][i]  # Get ID from 'ids' array
            score = filtered_results['distances'][0][i]  # Get score from 'distances' array
            # Retrieve text data from the filtered_results
            text = filtered_results['documents'][0][i]
            if not text:
                print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
            else:
                print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')

    except Exception as e:
        print(f"Error: {e}")
def main():
    try:
        # Delete old collection if it exists to avoid dimension mismatch errors
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "A collection for storing groceries",
                "hnsw:space": "cosine" # Correct way to set the metric
            },
            embedding_function=ef # Pass directly at the top level
        )
        print(f"Collection created: {collection_name}")
        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        ids =   [f"food_{index + 1}" for index, _ in enumerate(texts)]
        collection.upsert(
            documents=texts,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
            ids=ids
        )
        all_items= collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")
        
        query_term = 'fruits'
        query_term = is_query_list(query_term)
        perform_similarity_search(collection, query_term)

    except Exception as error:
        print(f"Error: {error}")


if __name__=="__main__":
    main()