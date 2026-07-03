template_facts = """Context information is below.
---------------------
{context_str}
---------------------
Based ONLY on the context information above, generate exactly 3 interesting and distinct facts about this person's career, education, or developed projects.
URGENT: If the context does not contain enough information to generate 3 facts, generate as many as possible based strictly on the context. Do not invent details.
Format the output as a clean numbered list (1., 2., 3.)."""

template_user_questions = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information, please answer this question: {query_str}

URGENT: Answer based ONLY on the provided context. If the information is not available or cannot be inferred from the context, politely respond with exactly: "I am sorry, but I don't know the answer to that based on the provided profile."
Make sure to formulate the response in a polite, professional, and highly formal corporate style."""

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
