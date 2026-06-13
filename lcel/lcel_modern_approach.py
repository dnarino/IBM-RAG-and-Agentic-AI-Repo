
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
model = ChatOpenAI()
parser= StrOutputParser() 

#2.Build your chain using the pipe operator

lcel_chain = prompt | model | parser

response = lcel_chain.invoke({"topic":"world Cup soccer 2016"})
print(response)