from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("Missing OPENAI_API_KEY. Did you create the .env file?")

model = ChatOpenAI()
prompt = PromptTemplate.from_template(
    "Tell me a short story about {topic} no more than 100 characters.."
)

legacy_chain = LLMChain(llm= model, prompt=prompt)
response = legacy_chain.run(topic="world Cup soccer 2016")
print(response)