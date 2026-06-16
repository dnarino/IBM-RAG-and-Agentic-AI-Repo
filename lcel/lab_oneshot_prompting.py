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
        "max_new_tokens": 150,
        "temperature": 0.1,
    }
llm = get_watsonx_llm(params)

# 2. Write a One-shot prompt
prompt_txt = """Return the latitude and longitude of the requested city in JSON format.

City: Paris
JSON: {"latitude": 48.8566, "longitude": 2.3522}

City: Bogota
JSON: """

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

# 1. One-shot prompt for formal email writing
formal_email_prompt = """Write a formal corporate email based on the provided topic.

Topic: Request a meeting with the marketing team to discuss the Q3 campaign.
Email:
Subject: Request for Meeting: Q3 Campaign Discussion

Dear Marketing Team,

I hope this email finds you well. I would like to request a meeting to discuss our upcoming\
Q3 marketing campaign. Please let me know your availability for next week so we can schedule\
a time to align our strategies.

Best regards,
[Your Name]

Topic: Announce that the office will be closed tomorrow because we wiil have been drinking beer \
and watching the colombia soccer match.
Email:
"""

# 2. One-shot prompt for simplifying technical concepts
technical_concept_prompt = """Simplify a technical concept into a brief summary of no more than \
50 words.

Topic: API (Application Programming Interface)
Concept: An API is a software bridge that allows different applications to talk to each other. \
Like a waiter in a restaurant, it takes your request to the system and brings the response back to you.

Topic: New Fabric AI model
Concept:
"""


promts= {
    "formal_email": formal_email_prompt,
    "technical_concept": technical_concept_prompt
}

for prompt_index, promt_text in promts.items():
    print(f"=== {prompt_index.upper()} RESPONSE ===")
    response = llm.invoke(promt_text)
    print(response)