
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("Missing OPENAI_API_KEY. Did you create the .env file?")

#1. Define the components
prompt = ChatPromptTemplate.from_template(
    "Tell me a short story about {topic} no more than 100 characters.."
)
model = ChatOpenAI(model="gpt-4o-mini")
parser= StrOutputParser() 

#2.Build your chain using the pipe operator

lcel_chain = prompt | model | parser

with get_openai_callback() as token_meter:
    # Everything executed inside this block is tracked!
    response = lcel_chain.invoke({"topic":"world Cup soccer 2016"})
   
    print("\n--- AI Response ---")
    print(response)
    
    print("\n--- Token Usage & Cost ---")
    print(f"Total Tokens: {token_meter.total_tokens}")
    print(f"Prompt Tokens (Input): {token_meter.prompt_tokens}")
    print(f"Completion Tokens (Output): {token_meter.completion_tokens}")
    print(f"Total Cost (USD): ${token_meter.total_cost:.5f}")