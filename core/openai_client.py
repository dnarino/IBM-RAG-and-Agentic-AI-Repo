import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Dict, Any

# Load environment variables from the .env file
load_dotenv()

def get_openai_chat(params: Dict[str, Any] = None) -> ChatOpenAI:
    """
    Initializes and returns a ChatOpenAI instance securely using environment variables.
    This function creates the Chat object so you can use it in LangChain pipelines (LCEL).
    """
    model_id = "gpt-4o-mini"

    # Note: OpenAI uses 'max_tokens' instead of 'max_new_tokens'
    default_params = {
        "max_tokens": 256,
        "temperature": 0.5,
        "model_kwargs": {
            "top_p": 0.2
        }
    }

    # If the user provides custom parameters, override the defaults
    if params:
        default_params.update(params)

    # Securely load credentials from environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables. Please add it to your .env file.")

    # Create the Chat instance using unpacking (**)
    chat_model = ChatOpenAI(
        model=model_id,
        api_key=api_key,
        **default_params
    )
    
    # Return the Chat object itself
    return chat_model
