from jinja2 import TemplateRuntimeError
import os
import sys

# Add the root project directory to the path so we can import our secure Watson client from Module 2!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.watson_client import get_watsonx_chat

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field


# pyrefly: ignore [missing-import]
from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID,OPENAI_MODEL_ID,GEMINI_MODEL_ID,DEEPSEEK_MODEL_ID

# 1. Define JSON output structure using Pydantic
class AIResponse(BaseModel):
    summary:str =Field(description="Summary of the user's message")
    sentiment:int= Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str= Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action:str=Field(description="Recommended action for the support rep")
    response: str = Field(description="Suggested response to the user")

json_outparser= JsonOutputParser(pydantic_object=AIResponse)

# 3. Initialize models securely
llama_LLM = get_watsonx_chat(model_id=LLAMA_MODEL_ID, params=PARAMETERS)
granite_LLM = get_watsonx_chat(model_id=GRANITE_MODEL_ID, params=PARAMETERS)
mistral_LLM = get_watsonx_chat(model_id=MISTRAL_MODEL_ID, params=PARAMETERS)

openai_LLM = ChatOpenAI(model=OPENAI_MODEL_ID, temperature=0.2)
gemini_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL_ID, temperature=0.2)
# DeepSeek (using the OpenAI client but pointing to DeepSeek's URL)

deepseek_LLM = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    model=DEEPSEEK_MODEL_ID,
    temperature=0.2
)


