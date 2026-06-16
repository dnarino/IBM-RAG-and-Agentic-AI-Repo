import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables from the .env file
load_dotenv()

# Initialize the LLM (Using OpenAI's gpt-4o-mini instead of IBM Watsonx)
# The ChatOpenAI class automatically uses the OPENAI_API_KEY environment variable.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=256
)

print("="*50)
print("LLM Initialized successfully")
print("="*50)


# ==========================================
# 1. Prompt Engineering Basics
# ==========================================

print("\n--- 1. Basic Prompt ---")
# Since ChatOpenAI expects messages, we can just invoke it with a string for basic prompting
prompt = "The wind is "
print(f"Prompt: {prompt}")
response = llm.invoke(prompt)
print(f"Response: {response.content}")

print("\n--- 2. Zero-shot Prompt ---")
prompt = """Classify the following statement as true or false: 
'The Eiffel Tower is located in Berlin.'

Answer:"""
print(f"Prompt:\n{prompt}")
response = llm.invoke(prompt)
print(f"Response: {response.content}")

print("\n--- 3. One-shot Prompt ---")
prompt = """Here is an example of translating a sentence from English to French:

English: “How is the weather today?”
French: “Comment est le temps aujourd'hui?”

Now, translate the following sentence from English to French:

English: “Where is the nearest supermarket?”
French:"""
print(f"Prompt:\n{prompt}")
response = llm.invoke(prompt)
print(f"Response: {response.content}")

print("\n--- 4. Few-shot Prompt ---")
prompt = """Here are a few examples of classifying emotions in statements:

Statement: 'I just won my first marathon!'
Emotion: Joy

Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration

Statement: 'My best friend is moving to another country.'
Emotion: Sadness

Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.'
Emotion:"""
print(f"Prompt:\n{prompt}")
response = llm.invoke(prompt)
print(f"Response: {response.content}")

print("\n--- 5. Chain-of-thought (CoT) Prompt ---")
prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
How many apples are there now?’

Break down each step of your calculation:"""
print(f"Prompt:\n{prompt}")
response = llm.invoke(prompt)
print(f"Response: {response.content}")

print("\n--- 6. Self-consistency Prompt ---")
prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

Provide three independent calculations and explanations, then determine the most consistent result."""
print(f"Prompt:\n{prompt}")
response = llm.invoke(prompt)
print(f"Response: {response.content}")


# ==========================================
# 2. Applications of Prompting with LCEL
# ==========================================
print("\n" + "="*50)
print("Applications with LangChain Expression Language (LCEL)")
print("="*50)

# Helper function to format prompts correctly for LCEL chains
def format_prompt(variables, prompt_template):
    return prompt_template.format(**variables)

print("\n--- A. Simple Prompt Template (Joke Generator) ---")
joke_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")
joke_chain = (
    RunnableLambda(lambda vars: format_prompt(vars, joke_template))
    | llm
    | StrOutputParser()
)
response = joke_chain.invoke({"adjective": "funny", "content": "chickens"})
print(f"Response (funny chickens): {response}")
response = joke_chain.invoke({"adjective": "sad", "content": "fish"})
print(f"Response (sad fish): {response}")

print("\n--- B. Text Summarization ---")
content_to_summarize = """
The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""
summarize_template = PromptTemplate.from_template("Summarize the following content in one sentence:\n{content}")
summarize_chain = (
    RunnableLambda(lambda vars: format_prompt(vars, summarize_template))
    | llm
    | StrOutputParser()
)
summary = summarize_chain.invoke({"content": content_to_summarize})
print(f"Summary: {summary}")

print("\n--- C. Question Answering ---")
qa_content = """
The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""
question = "Which planets in the solar system are rocky and solid?"
qa_template = PromptTemplate.from_template("""
Answer the {question} based on the {content}.
Respond "Unsure about answer" if not sure about the answer.

Answer:
""")
qa_chain = (
    RunnableLambda(lambda vars: format_prompt(vars, qa_template))
    | llm
    | StrOutputParser()
)
answer = qa_chain.invoke({"question": question, "content": qa_content})
print(f"Answer: {answer}")

print("\n--- D. Text Classification ---")
classification_text = "The concert last night was an exhilarating experience with outstanding performances by all artists."
categories = "Entertainment, Food and Dining, Technology, Literature, Music."
classification_template = PromptTemplate.from_template("""
Classify the following text into one of these categories: {categories}.

Text: {text}

Category:
""")
classification_chain = (
    RunnableLambda(lambda vars: format_prompt(vars, classification_template))
    | llm
    | StrOutputParser()
)
category = classification_chain.invoke({"text": classification_text, "categories": categories})
print(f"Category: {category}")

print("\n--- E. Code Generation (SQL) ---")
sql_description = """
Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
The table 'purchases' contains a column 'purchase_date'.
"""
sql_template = PromptTemplate.from_template("""
Generate an SQL query based on the following description:
{description}

SQL Query:
""")
sql_generation_chain = (
    RunnableLambda(lambda vars: format_prompt(vars, sql_template))
    | llm
    | StrOutputParser()
)
sql_query = sql_generation_chain.invoke({"description": sql_description})
print(f"SQL Query:\n{sql_query}")


print("\n--- F. Exercise 5: Product Review Analyzer ---")
review_template = PromptTemplate.from_template("""
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
""")
review_analysis_chain = (
    RunnableLambda(lambda vars: format_prompt(vars, review_template))
    | llm
    | StrOutputParser()
)

reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]

for i, review in enumerate(reviews):
    print(f"\n==== Review #{i+1} ====")
    result = review_analysis_chain.invoke({"review": review})
    print(result)

# Note: The role-playing interactive chat loop is not included here to prevent the script from blocking
# during execution. You can add it back if you want to run this script interactively.
