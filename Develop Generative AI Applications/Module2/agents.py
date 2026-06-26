from langchain_core.prompts import  PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool, tool


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

# Add tools
python_repl= PythonREPL()
python_calculator=Tool(
    name="Python Calculator",
    func=python_repl.run,
    description="Useful for when you need to perform calculations..."
)
@tool
def search_weather(location:str):
    """
        Search for the current weather in the specified location.
    """
    return f"The weather in {location} is currently sunny and 72°F."

tools= [python_calculator,search_weather]


# Create the ReAct agent prompt template
# The ReAct prompt needs to instruct the model to follow the thought-action-observation pattern

prompt_template = """
You are an agent who has access to the following tools:

{tools}

The available tools are: {tool_names}

To use a tool, please use the following format:
```
Thought: I need to figure out what to do
Action: tool_name
Action Input: the input to the tool
```

After you use a tool, the observation will be provided to you:
```
Observation: result of the tool
```

Then you should continue with the thought-action-observation cycle until you have enough information to respond to the user's request directly.
When you have the final answer, respond in this format:
```
Thought: I know the answer
Final Answer: the final answer to the original query
```

Remember, when using the Python Calculator tool,the input must be valid Python code.

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

agent =create_react_agent(
    llm=llama_LLM,
    tools=tools,
    prompt=prompt
)

agent_executor= AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

if __name__== "__main__":
    # Ask the agent a question that requires only calculation
    result = agent_executor.invoke({"input": "What is the square root of 256?"})
    print(result["output"])
    # Examples of different types of queries to test the agent
    queries = [
        "What's 345 * 789?",
        "Calculate the square root of 144",
        "What's the weather in Miami?",
        "If it's sunny in Chicago, what would be a good outdoor activity?",
        "Generate a list of prime numbers below 50 and calculate their sum"
    ]
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")    
        result = agent_executor.invoke({"input": query})
        print(f"\nFINAL ANSWER: {result['output']}")