import gradio as gr
import chromadb
import pandas as pd

# Connect to your Docker container
client = chromadb.HttpClient(host="localhost", port=8000)

def get_collections():
    try:
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception:
        return []

def show_collection_details(collection_name):
    if not collection_name:
        return "Please select a collection", pd.DataFrame()
    try:
        col = client.get_collection(name=collection_name)
        data = col.get(include=["documents", "metadatas"])
        
        df_dict = {
            "ID": data["ids"],
            "Document": data["documents"],
        }
        
        # Format metadata dictionary as a readable string
        meta_list = []
        for m in data.get("metadatas", []):
            meta_list.append(str(m) if m else "")
        df_dict["Metadata"] = meta_list
        
        df = pd.DataFrame(df_dict)
        summary = f"📦 Collection: {collection_name} | Total Records: {col.count()}"
        return summary, df
    except Exception as e:
        return f"Error: {e}", pd.DataFrame()

def run_semantic_search(collection_name, query_text):
    if not collection_name or not query_text:
        return pd.DataFrame()
    try:
        col = client.get_collection(name=collection_name)
        results = col.query(
            query_texts=[query_text],
            n_results=5
        )
        
        df_dict = {
            "ID": results["ids"][0],
            "Document": results["documents"][0],
            "Distance": [f"{d:.4f}" for d in results["distances"][0]]
        }
        
        meta_list = []
        for m in results.get("metadatas", [[]])[0]:
            meta_list.append(str(m) if m else "")
        df_dict["Metadata"] = meta_list
        
        return pd.DataFrame(df_dict)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def refresh_dropdown():
    return gr.Dropdown(choices=get_collections())

# Build Modern UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 ChromaDB Vector Database Viewer")
    
    with gr.Row():
        col_dropdown = gr.Dropdown(
            choices=get_collections(), 
            label="Select Collection", 
            interactive=True,
            value=get_collections()[0] if get_collections() else None
        )
        refresh_btn = gr.Button("🔄 Refresh Collections", size="sm")
        
    summary_text = gr.Markdown("Select a collection to view details")
    
    with gr.Tab("📋 View Collection Records"):
        table_output = gr.Dataframe(interactive=False)
        
    with gr.Tab("🔍 Test Semantic Query"):
        with gr.Row():
            query_input = gr.Textbox(placeholder="Enter search query here...", label="Search Query")
            search_btn = gr.Button("⚡ Search", variant="primary")
        search_results = gr.Dataframe(interactive=False)

    # Bind Events
    col_dropdown.change(
        fn=show_collection_details,
        inputs=[col_dropdown],
        outputs=[summary_text, table_output]
    )
    refresh_btn.click(
        fn=refresh_dropdown,
        outputs=[col_dropdown]
    )
    search_btn.click(
        fn=run_semantic_search,
        inputs=[col_dropdown, query_input],
        outputs=[search_results]
    )
    demo.load(
        fn=show_collection_details,
        inputs=[col_dropdown],
        outputs=[summary_text, table_output]
    )

if __name__ == "__main__":
    # Start Gradio interface
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
