# Suppress warnings
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

parameters = {
    "max_new_tokens": 256,
    "temperature": 0.2, 
}

# 2. Get the modularized LLM
granite_llm = get_watsonx_chat(model_id=model_id, params=parameters)

if __name__ == "__main__":
    print(f"Testing modular LangChain connection with model: {model_id}...\n")
    response = granite_llm.invoke(
        [
            SystemMessage(content="You are a helpful AI bot that assists a user \
                 in choosing the perfect book to read in one short sentence"),
            HumanMessage(content="I enjoy Technical and Hacker novels, what should I read?")
        ]

    )
    print(f"🤖 Response Text: {response.content}")
    
    # 3. Calculate and print cost using our modularized pricing utility!
    from core.pricing import print_execution_cost
    print_execution_cost(response, model_id)