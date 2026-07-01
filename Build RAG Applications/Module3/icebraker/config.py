template_facts="""
    generate 3 interesting facts about a person's career, education or developed projects
    use the {context_str} how context.
    URGENT: answer based only in the provided context.
    output:Requests detailed answers about the person
"""

template_user_questions="""
    Purpose: Answers spacific questions about a linkedIn profile
    Context: Uses the {context_str} placeholder for Linkedin data
    Query: Uses the {query_str} placeholder for the user's question
    URGENT:answer based only on the context, and to say "I don't know" if the information isn't available
"""

#Model Parameters
PARAMETERS={
    "max_new_tokens":256,
    "tempeture":0.2
}

# Model IDs
OPENAI_MODEL_ID = "gpt-4o-mini"

# Adjust these settings if needed
CHUNK_SIZE = 400  # Smaller chunks for more granular retrieval
SIMILARITY_TOP_K = 7  # Retrieve more chunks for more comprehensive answers

db_persistence_path="./chroma_db"
