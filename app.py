import os
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from crew import create_crew  # Local import of the crew setup

# Load environment variables
load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    # Check if GROQ_API_KEY is configured
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        await cl.Message(
            content="⚠️ **GROQ_API_KEY is not set!**\n\n"
                    "Please create a `.env` file in the root of the project (you can copy `.env.example`) "
                    "and set your `GROQ_API_KEY` before starting the crew."
        ).send()
    else:
        # Welcome message
        await cl.Message(
            content="🚀 **Chainlit + CrewAI Chatbot Initialized!**\n\n"
                    "Start chatting with the Llama 8b-powered conversational assistant below."
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Check for API key again
    if not os.environ.get("GROQ_API_KEY"):
        await cl.Message(
            content="❌ **Cannot execute.** Please configure `GROQ_API_KEY` in your `.env` file first."
        ).send()
        return

    # Inform the user that the chatbot is thinking
    status_msg = cl.Message(content="🤖 Thinking...")
    await status_msg.send()

    try:
        # Create the crew with the user's input
        crew = create_crew(message.content)
        
        # Execute the crew synchronously in a separate thread to keep the UI responsive
        result = await asyncio.to_thread(crew.kickoff)
        
        # Send the final response back to the UI
        await cl.Message(
            content=f"{result.raw}"
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ **An error occurred:** {str(e)}").send()
