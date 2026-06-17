import sys
import os

# Add parent directory to sys.path so 'core' module can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
params = {
        "max_new_tokens": 100,
    }
llm = get_watsonx_llm(params)

# 2. Write a Few-shot prompt
prompt_txt = """
    Here are few examples of classifying emotions in statements:

    Statement: 'I just won my first marathon!'
    Emotion: Joy
    
    Statement: 'I can't believe I lost my keys again.'
    Emotion: Frustration
    
    Statement: 'My best friend is moving to another country.'
    Emotion: Sadness
    
    Now, classify the emotion in the following statement:
    Statement: 'That movie was so scary I had to cover my eyes.'
    Emotion:
"""

# 3. Invoke and print!
"""
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
"""

## Starter code: provide your solutions in the TODO parts

# 1. One-shot prompt for formal email writing
tone_style_transfer_prompt = """
Translate the frustrated message into a polite, professional corporate response.Please check the next 2 examples

Frustrated: "This software is absolute garbage and crashes all the time! Fix it!"
Professional: "We apologize for the instability you are experiencing. Our team is actively \
investigating the crashes to improve performance."

Frustrated: "You charged me twice for no reason! Give me my money back right now!"
Professional: "We are sorry for the billing error. We are currently processing a refund for \
the duplicate charge."

response only the next :

Frustrated: "I've been waiting on hold for an hour, your customer service is a joke."
Professional:
"""

promts= {
    "formal_responses": tone_style_transfer_prompt,
}

for prompt_index, promt_text in promts.items():
    print(f"=== {prompt_index.upper()} RESPONSE ===")
    response = llm.invoke(promt_text)
    print(response)