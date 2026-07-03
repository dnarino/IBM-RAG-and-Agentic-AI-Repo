from llama_index.core import VectorStoreIndex, PromptTemplate
import config

def generate_initial_facts(index: VectorStoreIndex, llm) -> str:
    """Query index to extract 3 interesting facts based on template."""
    facts_prompt = PromptTemplate(config.template_facts)
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=facts_prompt,
        similarity_top_k=config.SIMILARITY_TOP_K
    )
    response = query_engine.query("Generate 3 interesting facts.")
    return str(response)

def answer_user_query(index: VectorStoreIndex, question: str, llm) -> str:
    """Query index to answer a specific user question formally."""
    qa_prompt = PromptTemplate(config.template_user_questions)
    query_engine = index.as_query_engine(
        llm=llm,
        text_qa_template=qa_prompt,
        similarity_top_k=config.SIMILARITY_TOP_K
    )
    response = query_engine.query(question)
    return str(response)
