import ssl
ssl.create_default_context=ssl._create_default_https_context

import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# 1. Connect to the remote Docker instance over HTTP
client = chromadb.HttpClient(host="localhost", port=8000)

# Check connection health
print(f"Server Heartbeat: {client.heartbeat()}")

collection_name = "football_collection"

def main():
    try:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description":"A collection for soccer players",
                "hnsw:space":"cosine"
            },
            embedding_function=ef
        )
        soccer_players = [
    {
        "id": "player_1",
        "name": "Lionel Messi",
        "position": "Forward",
        "age": 36,
        "club": "Inter Miami",
        "nationality": "Argentina",
        "skills": "Dribbling, playmaking, free kicks, close control, finishing",
        "goals": 821
    },
    {
        "id": "player_2",
        "name": "Cristiano Ronaldo",
        "position": "Forward",
        "age": 39,
        "club": "Al Nassr",
        "nationality": "Portugal",
        "skills": "Heading, athleticism, positioning, penalty kicks, finishing",
        "goals": 875
    },
    {
        "id": "player_3",
        "name": "Kevin De Bruyne",
        "position": "Midfielder",
        "age": 32,
        "club": "Manchester City",
        "nationality": "Belgium",
        "skills": "Passing range, vision, crossing, long shots, assists",
        "goals": 150
    },
    {
        "id": "player_4",
        "name": "Virgil van Dijk",
        "position": "Defender",
        "age": 32,
        "club": "Liverpool",
        "nationality": "Netherlands",
        "skills": "Tackling, aerial duels, leadership, positioning, strength",
        "goals": 50
    },
    {
        "id": "player_5",
        "name": "Erling Haaland",
        "position": "Forward",
        "age": 23,
        "club": "Manchester City",
        "nationality": "Norway",
        "skills": "Pace, strength, off-the-ball movement, clinical finishing",
        "goals": 230
    },
    {
        "id": "player_6",
        "name": "Alisson Becker",
        "position": "Goalkeeper",
        "age": 31,
        "club": "Liverpool",
        "nationality": "Brazil",
        "skills": "Shot-stopping, one-on-ones, distribution, positioning",
        "goals": 1
    }
        ]

        # List names of all collections currently available
        all_collections = client.list_collections()
        print([c.name for c in all_collections])

        # 1. Establish a connection to the existing collection
        collection = client.get_collection(name="football_collection")
        print("Before update:", collection.metadata)

        # 2. Update the descriptive tags (metadata) without changing the name
        collection.modify(
            metadata={
                "description": "Updated database tracking football players",
                "version": "2.0"
            }
        )
        print("After update:", collection.metadata)

        # Delete the entire collection and all of its contents permanently
        client.delete_collection(name="football_collection")
        print("💥 Collection erased successfully.")


    except Exception as error:
        print(f"{error}")

if __name__=="__main__":
    main()