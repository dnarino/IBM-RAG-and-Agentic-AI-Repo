import sys
import os

# Add parent directory to sys.path so 'core' module can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.watson_client import get_watsonx_llm

# 1. Get the model engine
params = {
        "max_new_tokens": 512,
    }
llm = get_watsonx_llm(params)

# 2. Write a One-shot prompt
prompt_txt = """
When I was 6, my sister was half of my age. Now I am 70, what age is my sister?
Provide three independent calculations and explanations, then determine the most \
consistent result.
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

