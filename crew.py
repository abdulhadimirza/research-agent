import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import litellm

# Configure LiteLLM to automatically drop unsupported parameters (doesn't look like its needed)
# litellm.drop_params = True

# Workaround for CrewAI injecting cache_breakpoint on unsupported models
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except ImportError:
    pass

# Load environment variables
load_dotenv()

def create_crew(user_query: str) -> Crew:
    # Initialize the LLM using Groq (defaulting to Llama 8b)
    groq_llm = LLM(
        model=os.environ.get("MODEL", "groq/llama3-8b-8192"),
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Define a chatbot agent
    chatbot = Agent(
        role="Friendly Chatbot Assistant",
        goal="Respond to the user's query in a helpful, conversational, and natural manner.",
        backstory="A polite, intelligent assistant designed to converse and help users with any questions.",
        verbose=True,
        llm=groq_llm
    )

    # Define a task to reply to the user's query
    chat_task = Task(
        description=f"Respond directly to this user message: {user_query}",
        expected_output="A helpful, conversational response to the user query.",
        agent=chatbot,
    )

    # Assemble the crew
    crew = Crew(
        agents=[chatbot],
        tasks=[chat_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
