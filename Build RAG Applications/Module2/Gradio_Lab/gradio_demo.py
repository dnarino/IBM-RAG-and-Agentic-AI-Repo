from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
import gradio as gr

# Correct import from your local core folder
from core.openai_client import get_openai_chat

# Initialize the OpenAI model with a larger max_tokens value to prevent responses from being cut off
openai_LLM = get_openai_chat(params={"max_tokens": 1024})

# Simulate a past chat history
past_history = InMemoryChatMessageHistory()
past_history.add_user_message("Hello my name is david and Im a soccer fanatic")
past_history.add_ai_message("Nice to meet you david, What can I do for you? ")

# return_messages=True is required when using ChatPromptTemplate & MessagesPlaceholder
memory = ConversationBufferMemory(chat_memory=past_history, return_messages=True)

# Build a ChatPromptTemplate with a MessagesPlaceholder for conversational history
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, expert soccer analyst."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)


# Connect the custom prompt template using the 'prompt' parameter
conversation = ConversationChain(
    llm=openai_LLM,
    verbose=True,
    memory=memory,
    prompt=prompt_template
)

# 💡 1. Define the response function at the module level so Gradio can see it
# 💡 stream_response yields updates in real-time.
# Gradio's ChatInterface automatically passes (message, history) to this function.
def stream_response(message: str, history: list):
    response_stream = conversation.stream({"input": message})
    current_response = ""
    for chunk in response_stream:
        if "response" in chunk:
            current_response += chunk["response"]
            yield current_response

# Define the Gradio ChatInterface
chat_application = gr.ChatInterface(
    fn=stream_response,
    title="⚽ SoccerChat Specialist",
    description="Ask any question related to soccer and the AI will answer it in real-time."
)

if __name__ == "__main__":
    # Launch the Gradio web server (omitting port so Gradio auto-finds a free one)
    chat_application.launch(server_name="127.0.0.1")

