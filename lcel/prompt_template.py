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
from core.watson_client import get_watsonx_chat

# 1. Get the model engine
params = {
        "max_new_tokens": 100,
        "temperature": 0.5
    }
llm = get_watsonx_chat(params)


# 2. Write a Template prompt
prompt_txt = """
Tell me a {adjective} joke about {content}
"""
prompt= PromptTemplate.from_template(prompt_txt)

## Create the chain WITHOUT the parser so we keep the metadata
lcel_chain= prompt | llm

# 3. Invoke and print!

print("Sending prompt to Watson...\n")
ai_message = lcel_chain.invoke({"adjective":"funny", "content":"chicken"})

# Extract the text (This does what StrOutputParser used to do)
print("\n--- AI Response ---")
print(ai_message.content)

# Extract the tokens directly from the response!
print("\n--- Token Usage (IBM Meta Llama) ---")
usage = ai_message.usage_metadata
if usage:
    print(f"Prompt Tokens (Input): {usage.get('input_tokens')}")
    print(f"Completion Tokens (Output): {usage.get('output_tokens')}")
    print(f"Total Tokens: {usage.get('total_tokens')}")
else:
    print("Token usage not returned by the API.")
