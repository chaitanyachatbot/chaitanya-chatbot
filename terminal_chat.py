from groq import Groq
from dotenv import load_dotenv
import os

# Load your API key from .env file
load_dotenv()

# Connect to Groq AI
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This is your chatbot's personality
SYSTEM_PROMPT = """You are a helpful, smart and friendly AI assistant.
You answer questions clearly and honestly.
You are similar to ChatGPT, Gemini and Claude."""

# This stores your conversation history
conversation_history = []

# Welcome message
print("=" * 50)
print("🤖 Welcome to My AI Chatbot!")
print("Type 'quit' to exit")
print("Type 'clear' to clear history")
print("=" * 50)

# Main chat loop - keeps running forever until you quit
while True:

    # Get user input
    user_input = input("\nYou: ").strip()

    # Skip if empty
    if not user_input:
        continue

    # Quit command
    if user_input.lower() == "quit":
        print("Bot: Goodbye! Have a great day! 👋")
        break

    # Clear history command
    if user_input.lower() == "clear":
        conversation_history = []
        print("Bot: Conversation history cleared! ✅")
        continue

    # Add your message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # Send to Groq AI and get response
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Free AI model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation_history  # Full conversation history
            ],
            max_tokens=1024,   # Max length of response
            temperature=0.7    # 0 = robotic, 1 = creative
        )

        # Extract the reply text
        bot_reply = response.choices[0].message.content

        # Add bot reply to history
        conversation_history.append({
            "role": "assistant",
            "content": bot_reply
        })

        # Print the reply
        print(f"\nBot: {bot_reply}")

    except Exception as e:
        print(f"\nBot: Sorry something went wrong! Error: {e}")