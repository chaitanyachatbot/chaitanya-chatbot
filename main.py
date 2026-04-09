from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="My AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful, smart and friendly AI assistant.
You answer questions clearly and honestly.
You are similar to ChatGPT, Gemini and Claude."""

conversations = {}

class Message(BaseModel):
    user_id: str
    message: str

class ClearChat(BaseModel):
    user_id: str

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/chat")
def chat(data: Message):
    user_id = data.user_id

    if user_id not in conversations:
        conversations[user_id] = []

    conversations[user_id].append({
        "role": "user",
        "content": data.message
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversations[user_id]
            ],
            max_tokens=1024,
            temperature=0.7
        )

        bot_reply = response.choices[0].message.content

        conversations[user_id].append({
            "role": "assistant",
            "content": bot_reply
        })

        return {
            "status": "success",
            "reply": bot_reply
        }

    except Exception as e:
        return {
            "status": "error",
            "reply": f"Sorry something went wrong! {str(e)}"
        }

@app.post("/clear")
def clear(data: ClearChat):
    if data.user_id in conversations:
        conversations[data.user_id] = []
    return {"status": "success", "message": "Chat cleared!"}

@app.get("/history/{user_id}")
def history(user_id: str):
    return {
        "user_id": user_id,
        "history": conversations.get(user_id, [])
    }