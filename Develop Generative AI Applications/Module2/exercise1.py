# Suppress warnings
from langchain_core.messages import AIMessage
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage

# Ensure the root directory is in the Python path so we can import 'core'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Import our modularized client!
from core.watson_client import get_watsonx_chat

# 1. Define model and parameters
model_id = 'ibm/granite-4-h-small'
model_id_opt='meta-llama/llama-3-3-70b-instruct'


if __name__ == "__main__":
    parameters = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_k": 100,         # Allow it to choose from the top 50 words
        "top_p": 0.9         # Allow a wide variety of probability 
    }

    # 2. Get the modularized LLM
    granite_llm = get_watsonx_chat(model_id=model_id, params=parameters)
    granite_llm_opt = get_watsonx_chat(model_id=model_id_opt, params=parameters)

    print(f"Creative writing")
    print(f"Testing modular LangChain connection with model: {model_id}...\n")
    response = granite_llm.invoke(
        [
            SystemMessage(content="You are a creative Poem Writer"),
            HumanMessage(content="Write a short poem about artificial intelligence"),
        ]
    )
    print(f"🤖 Response Text: {response.content}")
    # 3. Calculate and print cost using our modularized pricing utility!
    from core.pricing import print_execution_cost
    print_execution_cost(response, model_id)


    print(f"{'-'*15}")
    parameters = {
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_k": 10,         # Allow it to choose from the top 50 words
        "top_p": 0.4         # Allow a wide variety of probability 
    }

    # 2. Get the modularized LLM
    granite_llm = get_watsonx_chat(model_id=model_id, params=parameters)
    granite_llm_opt = get_watsonx_chat(model_id=model_id_opt, params=parameters)
    print(f"Factual questions")
    print(f"Testing modular LangChain connection with model: {model_id}...\n")
    response = granite_llm.invoke(
        [
            SystemMessage(content="Act as an Machine Learning Specialist"),
            HumanMessage(content="What are the key components of a neural network? \
                please use no more than 5 bulletpoints to explain"),
        ]
    )
    print(f"🤖 Response Text: {response.content}")
    # 3. Calculate and print cost using our modularized pricing utility!
    print_execution_cost(response, model_id)

   
    print(f"{'*' * 12}")
    parameters = {
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_k": 1,         # Allow it to choose from the top 50 words
        "top_p": 0.2         # Allow a wide variety of probability 
    }

    # 2. Get the modularized LLM
    granite_llm = get_watsonx_chat(model_id=model_id, params=parameters)
    granite_llm_opt = get_watsonx_chat(model_id=model_id_opt, params=parameters)
    print(f"Testing modular LangChain connection with model: {model_id_opt}...\n")
    response = granite_llm_opt.invoke(
        [
            SystemMessage(content="Act as an Machine Learning Specialist"),
            HumanMessage(content="What are the key components of a neural network? \
                please use no more than 5 bulletpoints to explain"),
        ]
    )
    print(f"🤖 Response Text: {response.content}")
    
    # 3. Calculate and print cost using our modularized pricing utility!
    print_execution_cost(response, model_id_opt)

    print(f"{'-'*15}")
    print(f"Instruction-following")
    print(f"Testing modular LangChain connection with model: {model_id}...\n")
    response = granite_llm.invoke(
        [
            SystemMessage(content="Act as an Time Management Expert"),
            HumanMessage(content="List 5 tips for effective time management"),
        ]
    )
    print(f"🤖 Response Text: {response.content}")
    # 3. Calculate and print cost using our modularized pricing utility!
    print_execution_cost(response, model_id)
    print(f"{'*' * 12}")
    print(f"Testing modular LangChain connection with model: {model_id_opt}...\n")
    response = granite_llm_opt.invoke(
        [
            SystemMessage(content="Act as an Time Management Expert"),
            HumanMessage(content="List 5 tips for effective time management"),
        ]
    )
    print(f"🤖 Response Text: {response.content}")
    
    # 3. Calculate and print cost using our modularized pricing utility!
    print_execution_cost(response, model_id_opt)