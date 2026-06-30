from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_classic.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import ChatWatsonx
from langchain_openai import ChatOpenAI

# Load environment variables (finds .env even if it's in a parent folder)
load_dotenv(find_dotenv())

def get_watsonx_llm():
    # Instruct models use the Chat endpoint on Watsonx
    model_id='meta-llama/llama-3-3-70b-instruct'

    parameters={
        GenParams.DECODING_METHOD: "greedy",  
        GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        "max_tokens": 256, # The chat endpoint looks for this specific parameter
        GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
    }

    credentials={
        # Use a fallback URL if WATSONX_URL is missing from the .env file
        "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "apikey": os.getenv("WATSONX_API_KEY")
    }

    # In modern langchain_ibm, chat models use ChatWatsonx instead of WatsonxLLM
    return ChatWatsonx(
        model_id=model_id,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params=parameters
    )

def get_openai_llm():
    """
    Creates an OpenAI Chat instance configured optimally for RAG chatbots.
    Recommended parameters:
    - temperature=0.0: Ensures factual, deterministic answers from the retrieved context.
    - model='gpt-4o-mini': Excellent balance of speed, cost, and reasoning capability for RAG.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
        api_key=os.getenv("OPENAI_API_KEY")
    )

def get_db():
    print("Loading database...")
    embeddings = HuggingFaceEmbeddings()
    # Load the local Chroma database
    base_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(base_dir, '../data/chroma_db')
    docsearch = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return docsearch

def qa_simple(llm):
    
    docsearch= get_db()
    
    # Create the retrieval chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        return_source_documents=False
    )
    query="what is mobile policy?"
    return qa.invoke(query)

def qa_prompt(llm):
    docsearch= get_db()

    prompt_template= """
        Use the information from the document to answer the \
        question at the end. If you don't know the answer, \
        just say that you don't know, definitely do not try to \
        make up an answer.
        {context}

        Question: {question} 
    """
    PROMPT = PromptTemplate(
        template= prompt_template,
        input_variables=["context","question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        chain_type_kwargs=chain_type_kwargs, 
        return_source_documents=False
    )
    query = "Can I eat in company vehicles?"
    return qa.invoke(query)
def qa_memory(llm):
    docsearch= get_db()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_message=True
    )
    qa= ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever= docsearch.as_retriever(),
        memory=memory,
        get_chat_history=lambda h:h,
        return_source_documents=False
    )
    history=[]
    query = "What is mobile policy"
    result = qa.invoke({"question": query}, {"chat_history": history})
    print(f"\n{'='*50}")
    print(f"🗣️  QUESTION (Turn 1): {query}")
    print(f"{'='*50}")
    print(f"🤖 ANSWER:\n{result['answer']}")
    print(f"{'='*50}\n")
    history.append((query, result["answer"]))
    
    query = "List points in it?"
    result = qa.invoke({"question": query}, {"chat_history": history})
    print(f"\n{'='*50}")
    print(f"🗣️  QUESTION (Turn 2): {query}")
    print(f"{'='*50}")
    print(f"🤖 ANSWER:\n{result['answer']}")
    print(f"{'='*50}\n")
    history.append((query, result["answer"]))
    
    query = "What is the aim of it?"
    result = qa.invoke({"question": query}, {"chat_history": history})
    print(f"\n{'='*50}")
    print(f"🗣️  QUESTION (Turn 3): {query}")
    print(f"{'='*50}")
    print(f"🤖 ANSWER:\n{result['answer']}")
    print(f"{'='*50}\n")
def qa_final(llm):

    docsearch= get_db()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_message = True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        memory = memory, 
        get_chat_history=lambda h : h, 
        return_source_documents=False
    )
    history = []
    print("\n" + "="*50)
    print("🤖 AI: Hello! I am your RAG Chatbot. Ask me anything about the documents!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("="*50 + "\n")
    
    turn = 1
    while True:
        query = input(f"🗣️  QUESTION (Turn {turn}): ")
        
        if query.lower() in ["quit", "exit", "bye"]:
            print(f"\n{'='*50}")
            print("🤖 ANSWER:\nGoodbye! Have a great day!")
            print(f"{'='*50}\n")
            break
            
        result = qa.invoke({"question": query, "chat_history": history})
        
        history.append((query, result["answer"]))
        
        print(f"{'='*50}")
        print(f"🤖 ANSWER:\n{result['answer']}")
        print(f"{'='*50}\n")
        
        turn += 1


def print_qa(response):
    print(f"\n{'='*50}")
    print(f"🗣️  QUESTION: {response['query']}")
    print(f"{'='*50}")
    print(f"🤖 ANSWER:\n{response['result']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 Welcome to the RAG Chatbot Lab!")
    print("="*50)
    
    print("\nSelect the LLM Provider:")
    print("1. IBM Watsonx (Meta Llama 3 70B)")
    print("2. OpenAI (GPT-4o-mini)")
    model_choice = input("Enter choice (1 or 2): ").strip()
    
    if model_choice == "2":
        llm = get_openai_llm()
    else:
        llm = get_watsonx_llm()
        
    print("\nSelect the QA Mode:")
    print("1. Simple QA (Hardcoded Question)")
    print("2. Prompt Template QA (Hardcoded Question)")
    print("3. Memory QA (Hardcoded 3-Turn Conversation)")
    print("4. Interactive Chat (Endless Loop)")
    qa_choice = input("Enter choice (1-4): ").strip()
    
    if qa_choice == "1":
        response = qa_simple(llm)
        print_qa(response)
    elif qa_choice == "2":
        response = qa_prompt(llm)
        print_qa(response)
    elif qa_choice == "3":
        qa_memory(llm)
    elif qa_choice == "4":
        qa_final(llm)
    else:
        print("Invalid choice. Exiting.")