from os import sched_rr_get_interval
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.watson_client import get_watsonx_llm

# 1. Get the model engine
llm = get_watsonx_llm()

# 2. Write a Zero-shot prompt
prompt_txt = """
Analyze the sentiment of the following customer review and classify it as POSITIVE, NEGATIVE, or NEUTRAL.
Review: I bought this laptop yesterday and the battery barely lasts 2 hours. Very disappointed
Sentiment:
"""

"""
Exercise 1

Analyze the sentiment of the following customer review and classify it as POSITIVE, NEGATIVE, or NEUTRAL.
Review: I bought this laptop yesterday and the battery barely lasts 2 hours. Very disappointed
Sentiment:


Translate the following English sentence into Spanish.
English: 'The artificial intelligence model is running perfectly on the server.'
Spanish:


Summarize the core concept of 'Agentic AI' in exactly one short sentence.
Summary:

Explain what a Python dictionary is to a 10-year-old child.
Explanation:

"""

# 3. Invoke and print!
print("=" * 40)
print("📝 PROMPT:")
print("-" * 40)
print(prompt_txt.strip())
print("=" * 40)
print("🤖 RESPONSE:")
print("-" * 40)
print("Sending prompt to Watson...\n")
response = llm.invoke(prompt_txt)
print(response)
print("=" * 40)


## Starter code: provide your solutions in the TODO parts

# 1. Prompt for Movie Review Classification
movie_review_prompt = """
Please depending the value of Movie Review IMDB , classify using the next parameters:
Parameters: <=0 Bad <=4 , >5 Medium <= 7 , >7 Good <=9 , <9 Excellent <=10
Movie: Superman 1
IMDB Review: 5
Result Clasification:
"""

# 2. Prompt for Climate Change Paragraph Summarization
climate_change_prompt = """
You are a Climate Change expert Journalist, search for the latest climate change news and give a summary \
no more than 50 words.
Summary:
"""

# 3. Prompt for English to Spanish Translation
translation_prompt = """
You are an expert multilingual translator, translate from English to Spanish the phrase 'What are you doing?'
Spanish:
"""

prompts = {
    "movie_review": movie_review_prompt,
    "climate_change": climate_change_prompt,
    "translation": translation_prompt
}

for prompt_type, prompt_text in prompts.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    print(f"Prompt: {prompt_text.strip()}\n")
    print("Generating response...")
    response = llm.invoke(prompt_text)
    print(f"Result:\n{response}\n")