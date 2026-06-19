# Suppress warnings
from langchain_core.messages import AIMessage
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
import sys

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

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
        "top_k": 3,         # Allow it to choose from the top 50 words
        "top_p": 0.2         # Allow a wide variety of probability 
    }
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

if __name__ == "__main__":
    
    print(f"Creative writing")
    print(f"Testing modular LangChain connection with model: {model_id}...\n")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a CallCenter Analyst , specialiced in find core issues"),
            MessagesPlaceholder("msgs")
        ]
    )
    
    chain = prompt_template | llama_LLM

    input_ = {
        "msgs": [
                HumanMessage(content="My internet is broken"),
                AIMessage(content="Have you tried restarting the router?"),
                HumanMessage(content="Yes, it is still blinking red")
            ]
        }

    response=chain.invoke(input_)
     # Debug the object:
    print("--- RAW AI MESSAGE OBJECT ---")
    print(response.model_dump_json(indent=2))
    print(f"🤖 Response Text: {response.content}")
    # 3. Calculate and print cost using our modularized pricing utility!
    from core.pricing import print_execution_cost
    print_execution_cost(response, model_id)

