import os
import sys
from dotenv import load_dotenv
import config
from llama_index.llms.openai import OpenAI

from modules.data_extraction import scrape_person_profile
from modules.data_processing import split_profile_text, create_vector_db, verify_embeddings
from modules.query_engine import generate_initial_facts, answer_user_query

load_dotenv()

def chatbot_loop(index, llm):
    """Provides interactive terminal chatbot loop to query the profile details."""
    print("\nYou can now ask questions about the person's profile. Type 'exit', 'quit', or 'bye' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye!")
            break
        if not query.strip():
            continue
        print("Bot is thinking...")
        response = answer_user_query(index, query, llm)
        print(f"Bot: {response}")

def main():
    name = input("Enter the name of the person you want to analyze: ")
    if not name.strip():
        print("Error: Name cannot be empty.")
        return
        
    openai_llm = OpenAI(
        model=config.OPENAI_MODEL_ID,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=config.PARAMETERS.get("tempeture", 0.2),
        max_tokens=config.PARAMETERS.get("max_new_tokens", 1200)
    )
    
    print("\n[1/4] Scraping profile details from public search...")
    profile_text = scrape_person_profile(name)
    if not profile_text:
        print("Failed to gather profile information.")
        return
        
    print("[2/4] Saving scraped data to disk...")
    os.makedirs("data", exist_ok=True)
    with open("data/scraped_profile.txt", "w", encoding="utf-8") as f:
        f.write(profile_text)
        
    print("[3/4] Chunking and indexing profile...")
    nodes = split_profile_text(profile_text)
    index = create_vector_db(nodes)
    verify_embeddings(index)
    
    print("[4/4] Generating initial facts...")
    facts = generate_initial_facts(index, openai_llm)
    
    print("\n" + "="*70)
    print(f"✨ 3 INTERESTING FACTS ABOUT {name.upper()}:")
    print("="*70)
    print(facts)
    print("="*70)
    
    chatbot_loop(index, openai_llm)

if __name__ == "__main__":
    main()
