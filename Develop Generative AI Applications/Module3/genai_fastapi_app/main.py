from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time

# pyrefly: ignore [missing-import]
from model import (
    llama_response, 
    granite_response, 
    mistral_response, 
    openai_response,
    gemini_response,
    AIResponse
)

app = FastAPI(title="AI Assistant API")

#Mount Static Files and Templates
app.mount("/static", StaticFiles(directory="static"),name="static")
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message:str
    model:str

@app.get("/")
async def index(request:Request):
    return templates.TemplateResponse(request=request, name="index.html")

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
        elif data.model == 'openai':
            result = openai_response(system_prompt, data.message)
        elif data.model == 'gemini':
            result = gemini_response(system_prompt, data.message)
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)