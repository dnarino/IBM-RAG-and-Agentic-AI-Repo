
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import our modularized client!
from core.watson_client import get_watsonx_chat

# 1. Define model and parameters
model_id='meta-llama/llama-3-3-70b-instruct'
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_k": 2,         # Strictly limit to the top 2 most likely words
    "top_p": 0.1        # Extremely low probability variance (highly factual)
}

# Set up the language model to use for chat interactions
llama_LLM = get_watsonx_chat(model_id=model_id, params=parameters)

# Create a new conversation history object
# This will store the back-and-forth messages in the conversation
history = ChatMessageHistory()

# Add an initial greeting message from the AI to the history
# This represents a message that would have been sent by the AI assistant
history.add_ai_message("hi!, Im a LLAMA Ai bot!!!")

# Add a user's question to the conversation history
# This represents a message sent by the user
history.add_user_message("what is the capital of France?")

prompt_template= ChatPromptTemplate.from_messages([
    ("system","You are a helpful AI assistant especiallized in Geography"),
    # . Add the dynamic Memory Placeholder!
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{question}")
])

print(history.messages)

if __name__ == "__main__":
    
    chain = prompt_template | llama_LLM
    response=chain.invoke({
        "chat_history":history.messages,
        "question":"And what is its population?"
    })
    print("--- RAW AI MESSAGE OBJECT ---")
    print(response.model_dump_json(indent=2))
    print(f"🤖 Response Text: {response.content}")


