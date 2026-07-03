import os
import uuid
import logging
from dotenv import load_dotenv
import gradio as gr
from llama_index.llms.openai import OpenAI
import config

from modules.data_extraction import scrape_person_profile
from modules.data_processing import split_profile_text, create_vector_db, verify_embeddings
from modules.query_engine import generate_initial_facts, answer_user_query

load_dotenv()

# Log configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Active user indexes
active_sessions = {}
openai_llm = OpenAI(
    model=config.OPENAI_MODEL_ID,
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=config.PARAMETERS.get("tempeture", 0.2),
    max_tokens=config.PARAMETERS.get("max_new_tokens", 256)
)

def process_profile(name: str):
    """Analyze profile, generate vector store index, and return initial facts and session ID."""
    if not name.strip():
        return "Please provide a valid name.", None
        
    try:
        profile_text = scrape_person_profile(name)
        if not profile_text:
            return "Failed to scrape profile information.", None
            
        nodes = split_profile_text(profile_text)
        index = create_vector_db(nodes)
        verify_embeddings(index)
        
        facts = generate_initial_facts(index, openai_llm)
        
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = index
        
        summary = f"Analysis complete for: {name}\n\n✨ 3 INTERESTING FACTS:\n{facts}"
        return summary, session_id
    except Exception as e:
        logger.error(f"Error in process_profile: {e}")
        return f"An error occurred: {e}", None

def chat_with_profile(session_id: str, message: str, history):
    """Perform RAG chat with the active session's vector store index."""
    if not session_id or session_id not in active_sessions:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "No active session. Please process a profile first."})
        return "", history
        
    if not message.strip():
        return "", history
        
    try:
        index = active_sessions[session_id]
        response = answer_user_query(index, message, openai_llm)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return "", history

def build_gui():
    """Create the tabbed Gradio interface blocks."""
    with gr.Blocks(title="LinkedIn Icebreaker RAG Bot") as demo:
        gr.Markdown("# 🤖 LinkedIn Icebreaker Bot")
        gr.Markdown("Search for a person, build a custom RAG index, extract facts, and chat.")
        
        session_id = gr.State(value="")
        
        with gr.Tab("1. Process Profile"):
            gr.Markdown("### Enter a name to query the web, scrape information, and index it.")
            name_input = gr.Textbox(label="Person Name to Analyze", placeholder="e.g., Donald Trump")
            process_btn = gr.Button("Analyze & Index", variant="primary")
            output_facts = gr.Textbox(label="Initial Facts Output", lines=12, interactive=False)
            
        with gr.Tab("2. Interactive Chat"):
            gr.Markdown("### Ask detailed questions about the indexed person.")
            chatbot = gr.Chatbot(label="Profile Assistant")
            chat_input = gr.Textbox(label="Your Question", placeholder="What are their career highlights?")
            send_btn = gr.Button("Send Question")
            
        process_btn.click(
            fn=process_profile,
            inputs=[name_input],
            outputs=[output_facts, session_id]
        )
        
        send_btn.click(
            fn=chat_with_profile,
            inputs=[session_id, chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )
        chat_input.submit(
            fn=chat_with_profile,
            inputs=[session_id, chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )
        
    return demo

if __name__ == "__main__":
    app = build_gui()
    # Run server locally on port 5000 to match lab guidelines
    app.launch(server_name="127.0.0.1", server_port=5000)
