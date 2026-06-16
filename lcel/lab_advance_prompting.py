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
Explain what a Python dictionary is to a 10-year-old child.
Explanation:
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
print("Sending prompt to Watson...")
response = llm.invoke(prompt_txt)
print(response)
