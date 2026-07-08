import ssl
import os
import chromadb
from chromadb.utils import embedding_functions

# WARNING: Disabling SSL verification is a temporary local workaround for corporate firewalls.
# Do not use this configuration in production/staging environments.
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
RESET_DATABASE = False  # Set to True only when you need to wipe/rebuild the index

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

collection_name = "library_collection"

def perform_advanced_search(collection)->None:
    try:
        #Similarity search for "magical fantasy adventure"
        print("\n1. Searching for magical fantasy adventure:")
        query_text = "magical fantasy adventure"
        similarity_results = collection.query(
            query_texts=query_text,
            n_results=3
        )
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
            similarity_results['ids'][0], similarity_results['documents'][0], similarity_results['distances'][0]
        )):
            metadata = similarity_results['metadatas'][0][i]
            print(f"{i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"Genre: {metadata['genre']}, Rating: {metadata['rating']}")
            print(f"Document: {document[:200]}...")
        #Filter books by genre (Fantasy or Science Fiction)
        filter_results= collection.get(
            where={"genre":{"$in":["Fantasy", "Science Fiction"]}}
        )
        print("\n2. Filtering books by genre (Fantasy or Science Fiction):")
        print(f"Found {len(filter_results['ids'])} books in these genres:")
        for i in range(len(filter_results['ids'])):
            title = filter_results['metadatas'][i]['title']
            genre = filter_results['metadatas'][i]['genre']
            rating = filter_results['metadatas'][i]['rating']
            print(f" - {title}: Genre: {genre}, Rating: {rating}")
        #Filter books by rating (4.0 or higher)
        filter_results= collection.get(
            where={"rating":{"$gte":4.0}}
        )
        print("\n3. Filtering books by rating (4.0 or higher):")
        print(f"Found {len(filter_results['ids'])} highly rated books:")
        for i in range(len(filter_results['ids'])):
            title = filter_results['metadatas'][i]['title']
            genre = filter_results['metadatas'][i]['genre']
            rating = filter_results['metadatas'][i]['rating']
            print(f" - {title} (Genre: {genre}) - Rating: {rating}")
    
        # === Combined Search: Similarity + Metadata Filtering ===
        # 6. highly-rated dystopian books with similarity search
        # Query:  teenage book
        # with filters: highly-rated and dystopian
        query_text = "teenage book"
        results = collection.query(
            query_texts=[query_text],
            n_results=3,
            where={
                "$and": [
                    {"rating": {"$gte": 4.0}},
                    {"genre": {"$in": ["Dystopian"]}}
                ]
            }
        )
        print("\n6. Combined Search (Highly-rated dystopian books):")
        print(f"Query: '{query_text}'")
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            print("No matching books found.")
        else:
            for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['distances'][0]
            )):
                metadata = results['metadatas'][0][i]
                print(f"{i+1}. {metadata['title']} ({doc_id}) - Distance: {distance:.4f}")
                print(f"Genre: {metadata['genre']}, Rating: {metadata['rating']}")
                print(f"Document: {document[:200]}...")
    except Exception as e:
        print(f"Error in advance search: {e}")
        
def main()->None:
    try:
        # Delete old collection if it exists and RESET_DATABASE is active
        if RESET_DATABASE:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
        #create collection
        collection= client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "A collection for storing books",
                "hnsw:space": "cosine" 
            },
            embedding_function=ef
        )

        books = [
            {
                "id": "book_1",
                "title": "The Great Gatsby",
                "author": "F. Scott Fitzgerald",
                "genre": "Classic",
                "year": 1925,
                "rating": 4.1,
                "pages": 180,
                "description": "A tragic tale of wealth, love, and the American Dream in the Jazz Age",
                "themes": "wealth, corruption, American Dream, social class",
                "setting": "New York, 1920s"
            },
            {
                "id": "book_2",
                "title": "To Kill a Mockingbird",
                "author": "Harper Lee",
                "genre": "Classic",
                "year": 1960,
                "rating": 4.3,
                "pages": 376,
                "description": "A powerful story of racial injustice and moral growth in the American South",
                "themes": "racism, justice, moral courage, childhood innocence",
                "setting": "Alabama, 1930s"
            },
            {
                "id": "book_3",
                "title": "1984",
                "author": "George Orwell",
                "genre": "Dystopian",
                "year": 1949,
                "rating": 4.4,
                "pages": 328,
                "description": "A chilling vision of totalitarian control and surveillance society",
                "themes": "totalitarianism, surveillance, freedom, truth",
                "setting": "Oceania, dystopian future"
            },
            {
                "id": "book_4",
                "title": "Harry Potter and the Philosopher's Stone",
                "author": "J.K. Rowling",
                "genre": "Fantasy",
                "year": 1997,
                "rating": 4.5,
                "pages": 223,
                "description": "A young wizard discovers his magical heritage and begins his education at Hogwarts",
                "themes": "friendship, courage, good vs evil, coming of age",
                "setting": "England, magical world"
            },
            {
                "id": "book_5",
                "title": "The Lord of the Rings",
                "author": "J.R.R. Tolkien",
                "genre": "Fantasy",
                "year": 1954,
                "rating": 4.5,
                "pages": 1216,
                "description": "An epic fantasy quest to destroy a powerful ring and save Middle-earth",
                "themes": "heroism, friendship, good vs evil, power corruption",
                "setting": "Middle-earth, fantasy realm"
            },
            {
                "id": "book_6",
                "title": "The Hitchhiker's Guide to the Galaxy",
                "author": "Douglas Adams",
                "genre": "Science Fiction",
                "year": 1979,
                "rating": 4.2,
                "pages": 224,
                "description": "A humorous space adventure following Arthur Dent across the galaxy",
                "themes": "absurdity, technology, existence, humor",
                "setting": "Space, various planets"
            },
            {
                "id": "book_7",
                "title": "Dune",
                "author": "Frank Herbert",
                "genre": "Science Fiction",
                "year": 1965,
                "rating": 4.3,
                "pages": 688,
                "description": "A complex tale of politics, religion, and ecology on a desert planet",
                "themes": "power, ecology, religion, politics",
                "setting": "Arrakis, distant future"
            },
            {
                "id": "book_8",
                "title": "The Hunger Games",
                "author": "Suzanne Collins",
                "genre": "Dystopian",
                "year": 2008,
                "rating": 4.2,
                "pages": 374,
                "description": "A teenage girl fights for survival in a brutal televised competition",
                "themes": "survival, oppression, sacrifice, rebellion",
                "setting": "Panem, dystopian future"
            },
        ]
        # Create comprehensive text documents for each book
        # These documents will be used for similarity search based on book themes, synopsis, and details
        book_documents=[]

        for book in books:
            document = (
                f"'{book['title']}' is a {book['genre']} novel written by {book['author']} and published in {book['year']}. "
                f"It contains {book['pages']} pages and holds an average rating of {book['rating']} stars. "
                f"Synopsis: {book['description']}. "
                f"Key themes explored in the book include {book['themes']}. "
                f"The narrative is set in {book['setting']}."
            )
            book_documents.append(document)
        ids = [book["id"] for book in books]
        collection.upsert(
            documents=book_documents,
            ids=ids,
            metadatas=[{
                "title":book["title"],
                "author":book["author"],
                "genre":book["genre"],
                "year":book["year"],
                "rating":book["rating"],
                "pages":book["pages"],
                "themes":book["themes"],
                "setting":book["setting"],
            }for book in books]
        )
        # Retrieving all items from the specified collection
        # The 'get' method fetches all records stored in the collection
        all_items = collection.get()
        # Logging the retrieved items to the console for inspection or debugging
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        #call quering
        perform_advanced_search(collection)
    except Exception as error:
        print(f"Error: {error}")

if __name__=="__main__":
    main()