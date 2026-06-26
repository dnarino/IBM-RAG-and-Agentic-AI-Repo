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
    deepseek_response,
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