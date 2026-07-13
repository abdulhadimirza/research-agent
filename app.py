import os
import asyncio
import chainlit as cl
from dotenv import load_dotenv
from src.crew import create_flow  # Local import of the flow setup
from crewai.agents.parser import AgentAction, AgentFinish

# Load environment variables
load_dotenv()

def make_step_callback(agent_role: str):
    """Factory to create step callbacks that bridge to the Chainlit UI."""
    def step_callback(step_output):
        if isinstance(step_output, AgentAction):
            step_name = f"[Agent] {agent_role} (Tool: {step_output.tool})"
            content = f"**Thought:** {step_output.thought}\n\n**Tool Input:** `{step_output.tool_input}`"
            if step_output.result:
                # Truncate tool results if they are excessively long for UI display
                res_str = str(step_output.result)
                if len(res_str) > 2000:
                    res_str = res_str[:2000] + "\n\n...(truncated for display)..."
                content += f"\n\n**Tool Output:**\n```markdown\n{res_str}\n```"
        elif isinstance(step_output, AgentFinish):
            step_name = f"[Finished] {agent_role}"
            content = f"**Thought:** {step_output.thought}\n\n**Final Answer draft:**\n{step_output.output}"
        else:
            step_name = f"[Step] {agent_role}"
            content = str(step_output)

        def update_ui():
            step = cl.Step(name=step_name, type="run")
            step.output = content
            return step.send()

        cl.run_sync(update_ui())
    
    return step_callback

def make_status_callback():
    """Factory to create status callbacks for the Flow steps."""
    def status_callback(msg: str):
        def update_ui():
            step = cl.Step(name="System Status", type="run")
            step.output = msg
            return step.send()
        cl.run_sync(update_ui())
    return status_callback

@cl.on_chat_start
async def on_chat_start():
    # Check if GROQ_API_KEY is configured
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        await cl.Message(
            content="**GROQ_API_KEY is not set!**\n\n"
                    "Please create a `.env` file in the root of the project (you can copy `.env.example`) "
                    "and set your `GROQ_API_KEY` before starting the flow."
        ).send()
    else:
        # Welcome message
        await cl.Message(
            content="**Chainlit + CrewAI Chatbot Initialized!**\n\n"
                    "Start chatting with the Multi-Agent, Tool-enabled assistant below."
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Check for API key again
    if not os.environ.get("GROQ_API_KEY"):
        await cl.Message(
            content="**Cannot execute.** Please configure `GROQ_API_KEY` in your `.env` file first."
        ).send()
        return

    # Inform the user that the chatbot is thinking
    status_msg = cl.Message(content="Thinking...")
    await status_msg.send()

    try:
        # Create the flow with the user's input and callbacks
        flow = create_flow(
            user_query=message.content, 
            step_callback_creator=make_step_callback,
            status_callback=make_status_callback()
        )
        
        # Execute the flow synchronously in a separate thread to keep the UI responsive
        result = await asyncio.to_thread(flow.kickoff)
        
        # Send the final response back to the UI
        await cl.Message(
            content=f"{result}"
        ).send()
    except Exception as e:
        await cl.Message(content=f"**An error occurred:** {str(e)}").send()
