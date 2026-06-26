"""
Exercise 7
Creating Your First LangChain Agent with Basic Tools

In this exercise, you'll build a simple agent that can help users with basic tasks using two custom tools. This exercise is a perfect starting point for understanding how LangChain agents work.

Instructions:

    Create two simple tools: A calculator and a text formatter.
    Set up a basic agent that can use these tools.
    Test the agent with straightforward questions.

"""

from langchain_core.prompts import  PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool, tool

@tool
def calculator(expression: str) -> str:
    """Useful for performing simple math calculations."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
@tool
def text_formatter(text: str) -> str:
    """Useful for formatting text. Input should be in the format 'format_type: your text'.
    Example formats are uppercase, lowercase, or titlecase.
    Example input: 'uppercase: hello world'"""
    try:
        if ":" not in text:
            return "Error: Input must be in format 'format_type: text'"
        
        # Split exactly once at the first colon
        format_type, content = text.split(":", 1)
        format_type = format_type.strip().lower()
        content = content.strip()
        
        if format_type == "uppercase":
            return content.upper()
        elif format_type == "lowercase":
            return content.lower()
        elif format_type == "titlecase":
            return content.title()
        else:
            return "Error: Unknown format type. Use uppercase, lowercase, or titlecase."
    except Exception as e:
        return f"Error: {e}"


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
tools =[calculator,text_formatter]

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

Remember, when using the calculator tool, the input must be a valid Python math expression.

--- Example ---
Question: Convert the phrase 'langchain is awesome' to uppercase.
Thought: The user wants to format text to uppercase. I should use the text_formatter tool.
Action: text_formatter
Action Input: uppercase: langchain is awesome
Observation: LANGCHAIN IS AWESOME
Thought: I have the formatted text and can answer the user.
Final Answer: The uppercase version is LANGCHAIN IS AWESOME.
--- End Example ---

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt= PromptTemplate.from_template(prompt_template)

agent=create_react_agent(
    llm=llama_LLM,
    tools=tools,
    prompt=prompt
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)



if __name__== "__main__":
    
    # queries
    queries = [
        "What is 25 + 63?", 
        "Can you convert 'hello world' to uppercase?",
        "Calculate 15 * 7", 
        "titlecase: langchain is awesome",
    ]
    for query in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")    
        result = agent_executor.invoke({"input": query})
        print(f"\nFINAL ANSWER: {result['output']}")