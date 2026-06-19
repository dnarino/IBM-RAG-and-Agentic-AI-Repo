"""
Pricing and token tracking utilities for LLM executions.
"""
from langchain_core.messages import AIMessage

# Official IBM pricing per 1 Million tokens
MODEL_PRICING = {
    "ibm/granite-4-h-small": {
        "input_cost": 0.0636,
        "output_cost": 0.265
    },
    "meta-llama/llama-3-3-70b-instruct": {
        # Note: Update with actual IBM pricing for Llama 3 if needed
        "input_cost": 1.80, 
        "output_cost": 1.80
    },
    
    # --- OpenAI Models ---
    "gpt-4o": {
        "input_cost": 5.00,
        "output_cost": 15.00
    },
    "gpt-4o-mini": {
        "input_cost": 0.15,
        "output_cost": 0.60
    },
    
    # --- Google Gemini Models ---
    "gemini-1.5-pro": {
        "input_cost": 3.50,
        "output_cost": 10.50
    },
    "gemini-1.5-flash": {
        "input_cost": 0.075,
        "output_cost": 0.30
    }
}

def print_execution_cost(response: AIMessage, model_id: str = "ibm/granite-4-h-small") -> None:
    """
    Extracts token usage from a LangChain AIMessage response and prints the calculated execution cost.
    
    Args:
        response (AIMessage): The response object returned by a Chat Model (.invoke)
        model_id (str): The ID of the model to look up accurate pricing.
    """
    token_usage = response.response_metadata.get("token_usage", {})
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    
    # Fetch pricing, default to 0 if model is unknown
    pricing = MODEL_PRICING.get(model_id, {"input_cost": 0.0, "output_cost": 0.0})
    
    input_cost = (input_tokens / 1_000_000) * pricing["input_cost"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_cost"]
    total_cost = input_cost + output_cost
    
    print("\n--- 💸 Execution Cost & Tokens ---")
    print(f"Model:         {model_id}")
    print(f"Input Tokens:  {input_tokens:<6} -> ${input_cost:.6f}")
    print(f"Output Tokens: {output_tokens:<6} -> ${output_cost:.6f}")
    print(f"Total Cost:    ${total_cost:.6f}")
    print("----------------------------------\n")
