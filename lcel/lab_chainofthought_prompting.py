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
        "max_new_tokens": 1000,
        "temperature":0.5
    }
llm = get_watsonx_llm(params)

# 2. Write a Few-shot prompt
prompt_txt = """
    Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
    How many apples are there now?

    Break down each step of your calculation
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

# 1. Prompt for decision-making process
decision_making_prompt = """

Consider the problem: I have a budget of $50.000 COP monthly for IA expenses,
I need to pay a model that could help me teaching me , copilot me develop code,
and a model wuth the capability for being used how LLM's in agents mainly langchain and crewai
My principal task are developing agentic systems(crew Ai , langchain) , I mainly develop in python
study AI , and Data engineer related courses. What company models do you recommend me and
give 3 model's name ?

Break down each step of your calculation.

"""
sandwich_making_promt="""
Consider the problem: I'm a deportist person today I burn 1200 calories, I eat healthy too.
I love the mediterranean food. Please give me the ingredients for a healthy sandwich.

Break down each step of you research.

"""


# 2. Prompt for explaining a process
##sandwich_making_prompt = ## TODO

prompts = {
    "decision_making": decision_making_prompt,
    "sandwich_making_promt":sandwich_making_promt

}

for prompt_type, promt_message in prompts.items():
    print(f"=== {prompt_type.upper()} RESPONSE ===")
    response = llm.invoke(promt_message)
    print(response)