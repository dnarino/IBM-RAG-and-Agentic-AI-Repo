
import os
import sys


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal




# Import our modularized client!
from core.watson_client import get_watsonx_chat

# 1. Define model and parameters
model_id='meta-llama/llama-3-3-70b-instruct'
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_k": 2,         # Strictly limit to the top 2 most likely words
    "top_p": 0.1        # Extremely low probability variance (highly factual)
}

# 2. Get the modularized LLM
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

#3. Setting the parser
#Define pydantic structure
class Movie(BaseModel):
    title:str=Field(description='this is the official movie title')
    director:str=Field(description='the name of the main movie director')
    year:int=Field(description='the official year of public movies launch')
    genre: Literal["Action", "Comedy", "Drama", "Sci-Fi", "Horror"] = Field(description='the type of genre of movie')

# Set up a parser + inject instructions into the prompt template.
output_parser =JsonOutputParser(pydantic_object=Movie)

# Get the formatting instructions for the output parser
# This generates guidance text that tells the LLM how to format its response
format_instructions= output_parser.get_format_instructions()


# Create a prompt template that includes:
# 1. Instructions for the LLM to answer the user's query
# 2. Format instructions to ensure the LLM returns properly structured data
# 3. The actual user query placeholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a JSON only assistance.
    Task: Generate info about the movie {movie_name} in JSON format
    \n\n{format_instructions}
    """),
    ("human", "{movie_name}")
])

if __name__ == "__main__":

    chat_prompt = chat_prompt.partial(format_instructions=format_instructions)
    
    # We must run the LLM first to get the raw message for our pricing utility
    llm_chain = chat_prompt | llama_LLM
    raw_response = llm_chain.invoke({'movie_name': 'rocky 4'})
    
    # Then we parse the raw message into JSON
    parsed_response = output_parser.invoke(raw_response)
    print(f"🤖 Parsed Output: {parsed_response}")

    # Calculate and print cost using our modularized pricing utility!
    from core.pricing import print_execution_cost
    print_execution_cost(raw_response, model_id)