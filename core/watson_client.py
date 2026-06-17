import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM, ChatWatsonx

# Load environment variables from the .env file
load_dotenv()

def get_watsonx_llm(params=None):
    """
    Initializes and returns a WatsonxLLM instance securely using environment variables.
    This function creates the LLM object so you can use it in LangChain pipelines (LCEL).
    """
    model_id = "meta-llama/llama-3-3-70b-instruct"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    # If the user provides custom parameters, override the defaults
    if params:
        default_params.update(params)

    # Securely load credentials from environment variables (no hardcoded keys!)
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = "https://us-south.ml.cloud.ibm.com"

    if not api_key or not project_id:
        raise ValueError("Missing WATSONX_API_KEY or WATSONX_PROJECT_ID in environment variables. Please add them to your .env file.")

    # Create the LLM instance
    granite_llm = WatsonxLLM(
        model_id=model_id,
        url=url,
        apikey=api_key,
        project_id=project_id,
        params=default_params
    )
    
    # Return the LLM object itself, NOT just the text response.
    # This allows us to use it anywhere in our LangChain pipelines!
    return granite_llm

def get_watsonx_chat(params=None):
    """
    Initializes and returns a ChatWatsonx instance securely using environment variables.
    This function creates the Chat object so you can use it in LangChain pipelines (LCEL).
    """
    model_id = "meta-llama/llama-3-3-70b-instruct"

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    # If the user provides custom parameters, override the defaults
    if params:
        default_params.update(params)

    # Securely load credentials from environment variables (no hardcoded keys!)
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = "https://us-south.ml.cloud.ibm.com"

    if not api_key or not project_id:
        raise ValueError("Missing WATSONX_API_KEY or WATSONX_PROJECT_ID in environment variables. Please add them to your .env file.")

    # Create the Chat instance
    chat_model = ChatWatsonx(
        model_id=model_id,
        url=url,
        apikey=api_key,
        project_id=project_id,
        params=default_params
    )
    
    # Return the Chat object itself
    # This allows us to use it anywhere in our LangChain pipelines!
    return chat_model
