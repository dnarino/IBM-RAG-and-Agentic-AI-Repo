
import os
import sys

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field



# Import OpenAI client
from core.openai_client import get_openai_chat

# Define the Pydantic schemas to hold the entire webpage content
class ContentSection(BaseModel):
    header: str = Field(description="The title or heading of this section")
    body_text: list[str] = Field(description="A list of the paragraphs in this section. Preserve the original text.")

class StructuredWebPage(BaseModel):
    page_title: str = Field(description="The overall title of the web page")
    sections: list[ContentSection] = Field(description="The webpage content broken down into chronological sections")

# Set up the JSON parser
output_parser = JsonOutputParser(pydantic_object=StructuredWebPage)
format_instructions = output_parser.get_format_instructions()

# Create a WebBaseLoader instance
loader = WebBaseLoader("https://webcatolicodejavier.org/")

# Set up the Chat Prompt Template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert data extractor. 
    Your task is to read the raw text of a website and reorganize ALL of the original text into a structured JSON format.
    Do NOT summarize the text. Preserve the original content, but organize it logically into headers and paragraph lists.
    \n\n{format_instructions}
    """),
    ("human", "Here is the raw website text to structure:\n\n{website_content}")
])

if __name__ == "__main__":
    
    # 1. Load the web data
    print("Scraping website...")
    web_data = loader.load()
    
    # 2. Get the OpenAI LLM (gpt-4o-mini)
    print("Initializing OpenAI model...")
    # We must increase max_tokens because returning an entire website takes a LOT of output tokens!
    llm = get_openai_chat(params={"max_tokens": 4096})
    
    # 3. Inject format instructions into the prompt
    chat_prompt = chat_prompt.partial(format_instructions=format_instructions)
    
    # 4. Create the chain
    llm_chain = chat_prompt | llm
    
    # 5. Invoke the chain passing the scraped text!
    print("Sending scraped text to LLM for structuring...")
    raw_response = llm_chain.invoke({'website_content': web_data[0].page_content})
    
    # 6. Parse the response into our Pydantic schema
    parsed_response = output_parser.invoke(raw_response)
    print(f"\n🤖 Structured JSON Output: \n{parsed_response}")

    # 7. Print the token usage and cost!
    from core.pricing import print_execution_cost
    print_execution_cost(raw_response, "gpt-4o-mini")

    # 8. Save the output to a properly formatted JSON file!
    import json
    with open("docs/test.json", "w", encoding="utf-8") as f:
        # json.dump automatically converts Python single quotes to JSON double quotes!
        json.dump(parsed_response, f, indent=4, ensure_ascii=False)
    print("\n✅ Successfully saved formatted JSON to docs/test.json")
