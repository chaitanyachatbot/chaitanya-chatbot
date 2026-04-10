from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import uuid
from datetime import datetime

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

FORMATTING RULES:
You are SMART about how you format responses based on what the user is asking:

WHEN TO USE FORMATTED RESPONSE (headings, bullets, emojis):
Use structured formatting ONLY for:
- Biography or information about a person (e.g. "who is Elon Musk", "tell me about APJ Kalam")
- Wikipedia style questions (e.g. "what is photosynthesis", "explain black holes")
- Lists and comparisons (e.g. "top 10 movies", "difference between X and Y")
- Step by step instructions (e.g. "how to cook biryani", "how to install Python")
- Facts and data (e.g. "facts about India", "history of Rome")
- Study notes or exam preparation

For these use:
## Headings for sections
* Bullet points for lists
**Bold** for important words
Numbered lists for steps
Emojis to make it engaging

WHEN TO USE NORMAL NATURAL RESPONSE:
Use plain natural flowing text for:
- Stories (e.g. "tell me a story", "write a story about a dragon")
- Jokes (e.g. "tell me a joke")
- Poems (e.g. "write a poem about rain")
- Casual conversation (e.g. "how are you", "what do you think")
- Creative writing of any kind
- Simple short questions (e.g. "what is 2+2")
- Moral stories / bedtime stories
- Any imaginative or creative request

For these write naturally like a human with no bullet points no headings just beautiful flowing text.

GENERAL RULES:
- Always match your tone to the question
- Be conversational and friendly
- Give complete and satisfying answers
- For stories write them fully with proper beginning middle and end
- For poems write them with proper rhythm and feel
- Never over-format a simple answer"""

CREATOR_REPLY = "I was created by Chaitanya Rama Narayana."

IDENTITY_TRIGGERS = [
    
    # ===== TRICKY INDIRECT QUESTIONS =====
    "what powers your", "what powers you",
    "under your hood", "under the hood",
    "running in your background", "running in background",
    "your engine", "ur engine",
    "what is your engine", "what is ur engine",
    "what drives your", "what drives you",
    "your core", "ur core",
    "what is your core", "what is ur core",
    "technology inside you", "tech inside you",
    "your brain made of", "ur brain made of",
    "makes you intelligent", "makes u intelligent",
    "gives you intelligence", "gives u intelligence",
    "processing unit", "your processing",
    "what is inside", "whats inside",

    # ===== SOUND / FEEL / SEEM TRICKS =====
    "you sound like", "u sound like",
    "you remind me of", "u remind me of",
    "you feel like", "u feel like",
    "you seem like", "u seem like",
    "you look like", "u look like",
    "your responses feel like", "ur responses feel like",
    "your style is similar", "ur style is similar",
    "similar to chatgpt", "similar to llama",
    "similar to gemini", "similar to claude",
    "i think you are llama", "i think u are llama",
    "i think you are chatgpt", "i think u are chatgpt",
    "i think you are gemini", "i think u are gemini",
    "am i right that you are", "am i right that u are",
    "guess what model", "tell me if you are",
    "confirm you are", "confirm u are",

    # ===== TECHNICAL PARAMETER TRICKS =====
    "token limit", "context window",
    "how many parameters", "billion parameters",
    "your temperature", "ur temperature",
    "max tokens", "top p value",
    "what model handles", "what model generates",
    "what generates your", "what generates ur",
    "what produces your", "what produces ur",
    "responsible for your responses",
    "who handles your responses",
    "what handles your responses",

    # ===== SYSTEM PROMPT TRICKS =====
    "system prompt", "your prompt",
    "ur prompt", "show me your prompt",
    "show me ur prompt", "reveal your prompt",
    "reveal ur prompt", "what instructions",
    "what were you told", "what were u told",
    "what instructions were you given",
    "initial prompt", "base prompt",
    "prompt template", "your configuration",
    "ur configuration", "your settings",
    "ur settings", "show configuration",

    # ===== REVERSE PSYCHOLOGY TRICKS =====
    "i already know you are", "i already know u are",
    "i already know you use", "i already know u use",
    "just confirm you are", "just confirm u are",
    "just say yes if you are", "just say yes if u are",
    "nod if you are", "blink if you are",
    "dont tell me who made you",
    "i know you are llama", "i know u are llama",
    "i know you are chatgpt", "i know u are chatgpt",
    "i know you are meta", "i know u are meta",
    "i know you use groq", "i know u use groq",
    "already know your api", "already know ur api",
    "i know your model", "i know ur model",

    # ===== WHO CREATED / MADE / BUILT =====
    "who created you", "who created u", "who created this",
    "who made you", "who made u", "who made this", "who made dis",
    "who built you", "who built u", "who built this",
    "who developed you", "who developed u", "who developed this",
    "who designed you", "who designed u", "who designed this",
    "who programmed you", "who programmed u", "who programmed this",
    "who coded you", "who coded u", "who coded this",
    "who wrote you", "who wrote u", "who wrote this",
    "who trained you", "who trained u", "who trained this",
    "who invented you", "who invented u", "who invented this",
    "who launched you", "who launched this",
    "who formed you", "who formed this",
    "who started you", "who started this",

    # ===== WHO ARE YOU =====
    "who are you", "who are u", "who r you", "who r u",
    "who is this", "who dis", "who is dis",
    "what are you", "what r you", "what r u",
    "what is this", "what is dis",
    "what am i talking to", "what am i chatting with",
    "what am i speaking to",
    "who am i talking to", "who am i chatting with",
    "who am i speaking to",

    # ===== NAME =====
    "what is your name", "what is ur name",
    "whats your name", "whats ur name",
    "what's your name", "what's ur name",
    "your name", "ur name",
    "tell me your name", "tell me ur name",
    "say your name", "say ur name",
    "what do i call you", "what do i call u",
    "what should i call you", "what should i call u",
    "do you have a name", "do u have a name",

    # ===== INTRODUCE YOURSELF =====
    "introduce yourself", "introduce urself",
    "tell me about yourself", "tell me about urself",
    "tell me about you", "tell me about u",
    "describe yourself", "describe urself",
    "about yourself", "about urself",
    "about you", "about u",
    "give me info about you", "give me info about u",
    "give me information about you",

    # ===== CREATOR / FOUNDER / OWNER =====
    "who is your creator", "who is ur creator",
    "who is your founder", "who is ur founder",
    "who is your owner", "who is ur owner",
    "who is your developer", "who is ur developer",
    "who is your maker", "who is ur maker",
    "who is your author", "who is ur author",
    "who is your programmer", "who is ur programmer",
    "who is your coder", "who is ur coder",
    "your creator", "ur creator",
    "your founder", "ur founder",
    "your owner", "ur owner",
    "your developer", "ur developer",
    "your maker", "ur maker",
    "your author", "ur author",
    "your programmer", "ur programmer",
    "your coder", "ur coder",
    "your master", "ur master",

    # ===== MADE BY / CREATED BY / BUILT BY =====
    "made by", "created by", "built by",
    "developed by", "designed by", "programmed by",
    "coded by", "trained by", "invented by",
    "owned by", "founded by", "launched by",
    "written by", "authored by",

    # ===== WHAT MODEL / AI / VERSION =====
    "what model", "which model",
    "what ai", "which ai",
    "what version", "which version",
    "what type of ai", "which type of ai",
    "what kind of ai", "which kind of ai",
    "what language model", "which language model",
    "what llm", "which llm",
    "what gpt", "which gpt",
    "what neural", "what transformer",
    "what algorithm",

    # ===== SPECIFIC AI NAMES =====
    "are you chatgpt", "r u chatgpt", "are u chatgpt",
    "is this chatgpt", "is it chatgpt",
    "are you gpt", "r u gpt", "are u gpt",
    "is this gpt", "is it gpt",
    "are you gpt4", "are you gpt3", "are you gpt 4", "are you gpt 3",
    "are you gemini", "r u gemini", "are u gemini",
    "is this gemini", "is it gemini",
    "are you claude", "r u claude", "are u claude",
    "is this claude", "is it claude",
    "are you llama", "r u llama", "are u llama",
    "is this llama", "is it llama",
    "are you llama2", "are you llama3", "are you llama 2", "are you llama 3",
    "are you meta", "r u meta", "are u meta",
    "is this meta", "is it meta",
    "are you openai", "r u openai", "are u openai",
    "is this openai", "is it openai",
    "are you groq", "r u groq", "are u groq",
    "is this groq", "is it groq",
    "are you bard", "r u bard", "are u bard",
    "is this bard", "is it bard",
    "are you copilot", "r u copilot", "are u copilot",
    "is this copilot", "is it copilot",
    "are you mistral", "r u mistral", "are u mistral",
    "is this mistral", "is it mistral",
    "are you perplexity", "r u perplexity",
    "are you bing", "r u bing",
    "are you alexa", "r u alexa",
    "are you siri", "r u siri",
    "are you cortana", "r u cortana",
    "are you google", "r u google",
    "are you anthropic", "r u anthropic",
    "are you huggingface", "r u huggingface",
    "are you falcon", "r u falcon",
    "are you vicuna", "r u vicuna",
    "are you alpaca", "r u alpaca",
    "are you cohere", "r u cohere",
    "are you palm", "r u palm",
    "are you deepmind", "r u deepmind",
    "are you a robot", "r u a robot", "are u a robot",
    "are you a bot", "r u a bot", "are u a bot",
    "are you real", "r u real", "are u real",
    "are you human", "r u human", "are u human",
    "are you an ai", "r u an ai", "are u an ai",
    "are you artificial intelligence",

    # ===== API =====
    "what api", "which api",
    "what api do you use", "what api do u use",
    "which api do you use", "which api do u use",
    "your api", "ur api",
    "what is your api", "what is ur api",
    "tell me your api", "tell me ur api",
    "what backend", "which backend",
    "your backend", "ur backend",
    "what is your backend",
    "groq api", "openai api", "meta api",
    "huggingface api", "anthropic api",

    # ===== CODE / PROGRAMMING =====
    "what code", "which code",
    "what coding language", "which coding language",
    "what programming language", "which programming language",
    "what language are you coded in",
    "what language are you written in",
    "what language are you programmed in",
    "how are you coded", "how are u coded",
    "how are you programmed", "how are u programmed",
    "your code", "ur code",
    "your source code", "ur source code",
    "show me your code", "show me ur code",
    "what is your code", "what is ur code",
    "are you python", "are you javascript",
    "are you java", "are you c++",
    "built with python", "coded in python",
    "written in python", "written in javascript",

    # ===== HTML / FRONTEND =====
    "your html", "ur html",
    "what html", "which html",
    "show me your html", "show me ur html",
    "what is your html", "what is ur html",
    "your frontend", "ur frontend",
    "what frontend", "which frontend",
    "your css", "ur css",
    "your javascript", "ur javascript",
    "your interface", "ur interface",
    "how is your interface made",
    "how is your website made",
    "how is your ui made",

    # ===== TECHNOLOGY / ARCHITECTURE =====
    "what technology", "which technology",
    "what tech", "which tech",
    "your technology", "ur technology",
    "your tech", "ur tech",
    "what is your technology",
    "powered by", "runs on", "built on",
    "based on", "uses what", "using what",
    "what framework", "which framework",
    "your framework", "ur framework",
    "your architecture", "ur architecture",
    "what architecture", "which architecture",
    "transformer model", "neural network",
    "what infrastructure", "which infrastructure",

    # ===== TRAINING / DATA =====
    "how were you trained", "how were u trained",
    "how are you trained", "how are u trained",
    "your training", "ur training",
    "your training data", "ur training data",
    "what data were you trained on",
    "what dataset", "which dataset",
    "your dataset", "ur dataset",
    "how were you made", "how were u made",
    "how were you built", "how were u built",
    "how were you created", "how were u created",
    "how are you made", "how are u made",
    "how are you built", "how are u built",
    "how are you created", "how are u created",
    "how do you work", "how do u work",
    "how does this work", "how does it work",
    "how does this ai work", "how does this bot work",

    # ===== COMPANY / ORGANIZATION =====
    "what company", "which company",
    "your company", "ur company",
    "what organization", "which organization",
    "your organization", "ur organization",
    "what firm", "which firm",
    "who owns you", "who owns u",
    "who runs you", "who runs u",
    "who controls you", "who controls u",
    "who manages you", "who manages u",
    "which company made you",
    "which company built you",
    "which company created you",
    "meta ai", "openai company", "anthropic company",
    "google ai", "microsoft ai",

    # ===== INTERNAL INFO =====
    "internal", "internal info",
    "internal information", "internal details",
    "show internal", "reveal internal",
    "what is inside you", "what is inside u",
    "your internals", "ur internals",
    "your details", "ur details",
    "your info", "ur info",
    "your information", "ur information",
    "give me your details", "give me ur details",
    "reveal yourself", "reveal urself",
    "expose yourself", "expose urself",

    # ===== SHORT FORMS / SLANG =====
    "whos ur creator", "whos ur maker",
    "whos ur owner", "whos ur dev",
    "ur dev", "ur coder",
    "who dev you", "who dev u",
    "who code you", "who code u",
    "who make you", "who make u",
    "who build you", "who build u",
    "who create you", "who create u",
    "who program you", "who program u",
    "who write you", "who write u",
    "who train you", "who train u",
    "who invent you", "who invent u",
    "wht r u", "wt r u",
    "wht are u", "wt are u",
    "wat r u", "wat are u",
    "hw r u made", "hw were u made",
    "hw r u built", "hw were u built",
    "hw r u created", "hw were u created",
    "hw u work", "hw do u work",

    # ===== HINDI / TELUGU STYLE ENGLISH =====
    "tumhe kisne banaya", "aapko kisne banaya",
    "kisne banaya", "kaun banaya",
    "kaun hai tera creator", "tera creator kaun hai",
    "nuvvu ela cheyabadav", "ninna yaru madidru",
    "enna model", "which model ra",
    "what model ra", "enti idi",
    "ela pani chestundi", "ela chestundi",

    # ===== ASKING ABOUT CHATBOT ITSELF =====
    "what is this chatbot", "what is this bot",
    "what is this app", "what is this application",
    "what is this website", "what is this site",
    "what is this tool", "what is this software",
    "tell me about this chatbot", "tell me about this bot",
    "tell me about this app",
    "info about this chatbot", "info about this bot",
    "details about this chatbot", "details about this bot",
    "this chatbot info", "this bot info",
    "explain this chatbot", "explain this bot",
    "explain this app",

    # ===== FOUNDATION / ORIGIN =====
    "your origin", "ur origin",
    "your foundation", "ur foundation",
    "your history", "ur history",
    "your background", "ur background",
    "your source", "ur source",
    "where do you come from", "where do u come from",
    "where are you from", "where r u from",
    "where were you created", "where were u created",
    "where were you made", "where were u made",
    "where were you built", "where were u built",
    "when were you created", "when were u created",
    "when were you made", "when were u made",
    "when were you built", "when were u built",
    "when were you launched", "when were u launched",
]

def is_identity_question(message: str) -> bool:
    # Clean message
    message_lower = message.lower().strip()
    message_clean = (message_lower
        .replace("?", "")
        .replace("!", "")
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace('"', "")
        .replace("-", " ")
        .strip()
    )

    # Check 1 — Direct keyword match (existing list)
    for trigger in IDENTITY_TRIGGERS:
        if trigger in message_clean:
            return True

    # Check 2 — Scoring system
    # If message contains enough suspicious words → block it
    identity_words = [
        "who", "what", "which", "how", "when", "where", "tell",
        "show", "reveal", "expose", "confirm", "explain", "describe",
        "you", "u", "ur", "your", "yourself",
    ]

    tech_words = [
        "model", "api", "ai", "llm", "gpt", "llama", "groq",
        "meta", "openai", "anthropic", "google", "microsoft",
        "created", "made", "built", "trained", "developed",
        "coded", "programmed", "designed", "invented",
        "creator", "founder", "owner", "developer", "maker",
        "name", "identity", "origin", "source", "base",
        "technology", "tech", "framework", "architecture",
        "backend", "frontend", "code", "html", "python",
        "server", "hardware", "gpu", "tpu", "compute",
        "limit", "quota", "rate", "tokens", "parameters",
        "prompt", "system", "configuration", "settings",
        "real", "true", "actual", "honest", "truth",
        "underlying", "beneath", "inside", "internal",
        "engine", "brain", "core", "power", "intelligence",
        "company", "organization", "firm", "team",
        "chatgpt", "gemini", "claude", "bard", "copilot",
        "mistral", "falcon", "vicuna", "alpaca", "cohere",
        "siri", "alexa", "cortana", "bing",
    ]

    # Count how many identity words appear
    identity_score = sum(1 for w in identity_words if w in message_clean.split())
    tech_score = sum(1 for w in tech_words if w in message_clean.split())

    # If message has identity word + tech word → block!
    if identity_score >= 1 and tech_score >= 1:
        return True

    # Check 3 — Very short suspicious messages
    short_triggers = [
        "who r u", "who ru", "wat r u", "wt r u",
        "whos this", "whos dat", "who dis",
        "ur model", "ur api", "ur name",
        "ur creator", "ur owner", "ur dev",
        "ur code", "ur html", "ur tech",
        "whos ur creator", "whos ur maker",
        "r u real", "r u ai", "r u bot",
        "r u human", "r u llama", "r u gpt",
        "r u meta", "r u google", "r u openai",
        "tell me", "show me", "reveal",
    ]

    for trigger in short_triggers:
        if trigger in message_clean:
            return True

    return False

all_chats = {}

class Message(BaseModel):
    user_id: str
    chat_id: str
    message: str

class NewChat(BaseModel):
    user_id: str
    chat_name: str
    private: bool = False
    password: str = ""

class DeleteChat(BaseModel):
    user_id: str
    chat_id: str

class AccessChat(BaseModel):
    user_id: str
    chat_id: str
    password: str = ""

def get_user_chats(user_id):
    if user_id not in all_chats:
        all_chats[user_id] = {}
    return all_chats[user_id]

@app.get("/")
def home():
    return FileResponse("index.html")

@app.post("/new_chat")
def new_chat(data: NewChat):
    chats = get_user_chats(data.user_id)
    chat_id = str(uuid.uuid4())[:8]
    chats[chat_id] = {
        "name": data.chat_name,
        "messages": [],
        "private": data.private,
        "password": data.password if data.private else "",
        "created_at": datetime.now().strftime("%d %b %Y %I:%M %p")
    }
    return {"status": "success", "chat_id": chat_id}

@app.get("/get_chats/{user_id}")
def get_chats(user_id: str):
    chats = get_user_chats(user_id)
    result = []
    for chat_id, chat in chats.items():
        result.append({
            "chat_id": chat_id,
            "name": chat["name"],
            "private": chat["private"],
            "created_at": chat["created_at"],
            "message_count": len(chat["messages"])
        })
    return {"chats": result}

@app.post("/access_chat")
def access_chat(data: AccessChat):
    chats = get_user_chats(data.user_id)
    if data.chat_id not in chats:
        return {"status": "error", "message": "Chat not found!"}
    chat = chats[data.chat_id]
    if chat["private"]:
        if data.password != chat["password"]:
            return {"status": "error", "message": "Wrong password!"}
    return {
        "status": "success",
        "messages": chat["messages"],
        "name": chat["name"]
    }

@app.post("/chat")
def chat(data: Message):
    chats = get_user_chats(data.user_id)
    if data.chat_id not in chats:
        return {"status": "error", "reply": "Chat not found!"}

    chat = chats[data.chat_id]

    # ===== IDENTITY CHECK =====
    # Intercept BEFORE sending to AI
    # AI never sees identity questions!
    if is_identity_question(data.message):
        chat["messages"].append({
            "role": "user",
            "content": data.message
        })
        chat["messages"].append({
            "role": "assistant",
            "content": CREATOR_REPLY
        })
        return {"status": "success", "reply": CREATOR_REPLY}

    # Normal question — send to AI
    chat["messages"].append({
        "role": "user",
        "content": data.message
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat["messages"]
            ],
            max_tokens=2048,
            temperature=0.8
        )

        bot_reply = response.choices[0].message.content
        chat["messages"].append({
            "role": "assistant",
            "content": bot_reply
        })

        return {"status": "success", "reply": bot_reply}

    except Exception as e:
        return {"status": "error", "reply": f"Error: {str(e)}"}

@app.post("/delete_chat")
def delete_chat(data: DeleteChat):
    chats = get_user_chats(data.user_id)
    if data.chat_id in chats:
        del chats[data.chat_id]
    return {"status": "success", "message": "Chat deleted!"}
