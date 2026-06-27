
import os
import sys

# Add the root project directory to the path so we can import our secure Watson client from Module 2!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.watson_client import get_watsonx_chat

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field


# pyrefly: ignore [missing-import]
from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID,OPENAI_MODEL_ID,GEMINI_MODEL_ID

# 1. Define JSON output structure using Pydantic
class AIResponse(BaseModel):
    summary:str =Field(description="Summary of the user's message")
    sentiment:int= Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str= Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action:str=Field(description="Recommended action for the support rep")
    response: str = Field(description="Suggested response to the user")

json_parser= JsonOutputParser(pydantic_object=AIResponse)

# 3. Initialize models securely
llama_LLM = get_watsonx_chat(model_id=LLAMA_MODEL_ID, params=PARAMETERS)
granite_LLM = get_watsonx_chat(model_id=GRANITE_MODEL_ID, params=PARAMETERS)
mistral_LLM = get_watsonx_chat(model_id=MISTRAL_MODEL_ID, params=PARAMETERS)

openai_LLM = ChatOpenAI(model=OPENAI_MODEL_ID, temperature=0.2)
gemini_LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL_ID, temperature=0.2)

llama_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}\n{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

granite_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

mistral_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{format_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

# ---------------------------------------------------------
# Modern Chat APIs (OpenAI, Gemini, DeepSeek)
# ---------------------------------------------------------
# Unlike Llama or Mistral, modern API providers don't require you to write out raw text tags 
# like <|begin_of_text|>. Instead, LangChain's ChatPromptTemplate handles the formatting automatically!

chat_template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n{format_prompt}"),
    ("user", "{user_prompt}")
])

# We can reuse the exact same template for all three!
openai_template = chat_template
gemini_template = chat_template

# 5. Build the Chain
def get_ai_response(model, template, system_prompt, user_prompt) -> AIResponse:
    chain = template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt, 
        'user_prompt': user_prompt, 
        'format_prompt': json_parser.get_format_instructions()
    })

def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_LLM, llama_template, system_prompt, user_prompt)

def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_LLM, granite_template, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_LLM, mistral_template, system_prompt, user_prompt)

def openai_response(system_prompt, user_prompt):
    return get_ai_response(openai_LLM, openai_template,system_prompt, user_prompt)

def gemini_response(system_prompt, user_prompt):
    return get_ai_response(gemini_LLM, gemini_template,system_prompt, user_prompt)
