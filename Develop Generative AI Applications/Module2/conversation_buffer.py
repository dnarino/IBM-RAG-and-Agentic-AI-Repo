
from langchain_core.chat_history import InMemoryChatMessageHistory
# Import ConversationBufferMemory from langchain.memory module
from langchain_classic.memory import ConversationBufferMemory
# Import ConversationChain from langchain.chains module
from langchain_classic.chains import ConversationChain
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

# 1. Create a raw history object and load your past conversation into it!
past_history = InMemoryChatMessageHistory()
past_history.add_user_message("Hello, my name is David and I am an engineer.")
past_history.add_ai_message("Nice to meet you, John! How can I help you?")

# 2. Give that pre-loaded history to the BufferMemory
memory = ConversationBufferMemory(chat_memory=past_history)

# Create a conversation chain with the following components:
conversation = ConversationChain(
    # The language model to use for generating responses
    llm=llama_LLM,
    
    # Set verbose to True to see the full prompt sent to the LLM, including memory contents
    verbose=True,
    
    # Initialize with ConversationBufferMemory that will:
    # - Store all conversation turns (user inputs and AI responses)
    # - Append the entire conversation history to each new prompt
    # - Provide context for the LLM to generate contextually relevant responses
    memory=memory
)


if __name__ == "__main__":
    response = conversation.invoke(input="What this IBM's courses in coursera are too outdated")
    print("\n🤖 AI Response:")
    print(response["response"])
    
    


