
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import format_instructions
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser

import os
from dotenv import load_dotenv

load_dotenv()
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("Missing OPENAI_API_KEY. Did you create the .env file?")

#1. Initialize the specialized list parser
list_parser = CommaSeparatedListOutputParser()

#2. Build prompt and inject the parser's specific formatting rules

prompt = ChatPromptTemplate.from_template(
    "list the 3 legendary players who played for national soccer selection of {national_selection} .\n{format_instructions}"
)

prompt = prompt.partial(format_instructions=list_parser.get_format_instructions())

#3. create the model 

model = ChatOpenAI(model="gpt-4o-mini")

#4. Chain them together using LCEL

soccer_list_chain = prompt | model | list_parser

#5. invoque the chain

with get_openai_callback() as token_meter:
    # Everything executed inside this block is tracked!
    for chunk in soccer_list_chain.stream({"national_selection":"Colombia"}):
        print(chunk , end="", flush=True)
    print("\n--- Token Usage & Cost ---")
    print(f"Total Tokens: {token_meter.total_tokens}")
    print(f"Prompt Tokens (Input): {token_meter.prompt_tokens}")
    print(f"Completion Tokens (Output): {token_meter.completion_tokens}")
    print(f"Total Cost (USD): ${token_meter.total_cost:.5f}")

print(type(soccer_list_chain)) 
print(soccer_list_chain)