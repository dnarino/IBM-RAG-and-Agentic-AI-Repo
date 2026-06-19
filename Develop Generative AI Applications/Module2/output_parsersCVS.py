


import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
import sys


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

# Ensure the root directory is in the Python path so we can import 'core'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ['ANONYMIZED_TELEMETRY'] = 'False'



# Import our modularized client!
from core.watson_client import get_watsonx_chat

# 1. Define model and parameters
model_id='meta-llama/llama-3-3-70b-instruct'
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_k": 5,         # Allow it to choose from the top 50 words
    "top_p": 0.2         # Allow a wide variety of probability 
}

# 2. Get the modularized LLM
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

#3. Setting the parser
#define pydantic structure
class Joke(BaseModel):
    setup:str=Field(description='question to set up a joke')
    punchline:str=Field(description='answer to solve the joke')

# Set up a parser + inject instructions into the prompt template.
output_parser =JsonOutputParser(pydantic_object=Joke)
# Get the formatting instructions for the output parser
# This generates guidance text that tells the LLM how to format its response
format_instructions= output_parser.get_format_instructions()


# Create a prompt template that includes:
# 1. Instructions for the LLM to answer the user's query
# 2. Format instructions to ensure the LLM returns properly structured data
# 3. The actual user query placeholder



prompt= PromptTemplate(
    template="Answer the user query .\n{format_instructions}\n {query}",
    input_variables=["query"],
    partial_variables={'format_instructions':format_instructions}

)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user query \n\n{format_instructions}"),
    ("human", "{query}")
])


if __name__ == "__main__":
    joke_query = "Tell me a joke."
  
    print(f"Testing modular LangChain connection with model: {model_id}...\n")
    
    # The full chain returns a dictionary, which loses the usage metadata needed for pricing.
    # To keep both, we can run the LLM first, print the cost, and then parse the output.
    llm_chain = prompt | llama_LLM
    raw_response = llm_chain.invoke({'query': joke_query})
    
    parsed_response = output_parser.invoke(raw_response)
    print(f"🤖 Parsed Output: {parsed_response}")
    
    # 3. Calculate and print cost using our modularized pricing utility!
    from core.pricing import print_execution_cost
    print_execution_cost(raw_response, model_id)

    #@Using ChatPromptTemplate 
    """
    This approach is highly recommended for modern models like LLaMA 3, 
    as they are specifically fine-tuned to distinguish between system rules and human inputs
    """  
    chat_prompt = chat_prompt.partial(format_instructions=format_instructions)
    chain = chat_prompt | llama_LLM | output_parser
    parsed_response = chain.invoke({'query': joke_query})
    print(f"🤖 Parsed Output: {parsed_response}")