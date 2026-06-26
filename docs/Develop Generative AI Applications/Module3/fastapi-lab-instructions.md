# Build Your First GenAI Application The Right Way (FastAPI Edition)

Build a smart AI-powered web application using **FastAPI** and cutting-edge language models! This hands-on project will teach you to create intelligent applications that generate structured responses and leverage enterprise-grade AI tools. You'll learn to integrate watsonx.ai, implement JSON parsing via Pydantic, and engineer prompts that deliver consistent, actionable results.

## Why FastAPI?

FastAPI is a modern, high-performance web framework for building APIs with Python. It is an excellent choice for AI applications because:
1. **Asynchronous by Default:** LLM API calls take time. FastAPI handles these delays gracefully without freezing your server.
2. **Native Pydantic Integration:** LangChain's structured output parsers rely on Pydantic. FastAPI also uses Pydantic for data validation, allowing us to seamlessly reuse our AI schemas for our Web API!

---

## Step 1: Setting Up Your Environment

First, let's create a new directory for this project and install the required packages. Open your terminal and run:

```bash
mkdir -p "Develop Generative AI Applications/Module3/genai_fastapi_app"
cd "Develop Generative AI Applications/Module3/genai_fastapi_app"

# Install FastAPI, Uvicorn (the web server), and LangChain dependencies
uv pip install fastapi uvicorn pydantic langchain-ibm langchain python-dotenv
```

---

## Step 2: The Model Configuration (`config.py`)

We need a central place to store our model IDs and settings. Create a file named `config.py`:

```python
# config.py
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Model parameters (We want deterministic, concise answers)
PARAMETERS = {
    "max_new_tokens": 256,
    "temperature": 0.2,
}

# Model IDs
LLAMA_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"
```

---

## Step 3: The AI Logic (`model.py`)

Now we will configure LangChain. We will import the secure `watson_client.py` from Module 2 to handle our API keys automatically! Create a file named `model.py`:

```python
# model.py
import sys
import os

# Add the root project directory to the path so we can import our secure Watson client from Module 2!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from core.watson_client import get_watsonx_chat

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID

# 1. Define JSON output structure using Pydantic
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action: str = Field(description="Recommended action for the support rep")
    response: str = Field(description="Suggested response to the user")

# 2. JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# 3. Initialize models securely
llama_llm = get_watsonx_chat(model_id=LLAMA_MODEL_ID, params=PARAMETERS)
granite_llm = get_watsonx_chat(model_id=GRANITE_MODEL_ID, params=PARAMETERS)
mistral_llm = get_watsonx_chat(model_id=MISTRAL_MODEL_ID, params=PARAMETERS)

# 4. Prompt Templates (Different models require different formatting tags!)
llama_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}\n{format_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

granite_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

mistral_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{format_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

# 5. Build the Chain
def get_ai_response(model, template, system_prompt, user_prompt) -> AIResponse:
    chain = template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt, 
        'user_prompt': user_prompt, 
        'format_prompt': json_parser.get_format_instructions()
    })

def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, llama_template, system_prompt, user_prompt)

def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_llm, granite_template, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_llm, mistral_template, system_prompt, user_prompt)
```

---

## Step 4: The Web Application (`main.py`)

Now we tie it all together with FastAPI! Create a file named `main.py`:

```python
# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

from model import llama_response, granite_response, mistral_response, AIResponse

app = FastAPI(title="AI Assistant API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define the incoming Request schema
class ChatRequest(BaseModel):
    message: str
    model: str

# 1. Serve the Frontend HTML
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 2. Handle the AI Chat API
@app.post("/generate", response_model=AIResponse)
async def generate(data: ChatRequest):
    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."
    
    try:
        # Notice we are using standard Pydantic models for request validation!
        if data.model == 'llama':
            result = llama_response(system_prompt, data.message)
        elif data.model == 'granite':
            result = granite_response(system_prompt, data.message)
        elif data.model == 'mistral':
            result = mistral_response(system_prompt, data.message)
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

---

## Step 5: The Frontend UI (HTML/CSS/JS)

FastAPI serves our HTML from a `templates` directory, and CSS/JS from a `static` directory.

Run these commands in your terminal to set up the directories and download the frontend files:

```bash
mkdir templates static

# Download HTML
wget -O templates/index.html "https://gist.githubusercontent.com/tenzinmigmar/5f69c0d4520e5c942ba6e54f86f3db17/raw/index.html"

# Download JS
wget -O static/script.js "https://gist.githubusercontent.com/tenzinmigmar/0168709391266a8d8da7936f1a866c71/raw/95f4f4e1a1966b3f5183dd2f822cfcfd08d2238a/script.js"

# Download CSS
wget -O static/styles.css "https://gist.githubusercontent.com/tenzinmigmar/278575598f79a4940993a1fc8640a60a/raw/24eda98885e854b01b4a46d1756112e91d3acc10/styles.css"
```

---

## Step 6: Testing the Application!

To run your new FastAPI application, execute:

```bash
python main.py
```
*(Or use `uvicorn main:app --reload`)*

Once running:
1. Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to see and use the Chat UI!
2. Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to see FastAPI's automatic, beautiful API documentation (Swagger UI), powered directly by your LangChain Pydantic models!
