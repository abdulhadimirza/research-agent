import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

# Workaround for CrewAI injecting cache_breakpoint on unsupported models
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except ImportError:
    pass

# Load environment variables
load_dotenv()

@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web for a query and return titles and URLs of top results."""
    try:
        with DDGS() as ddgs:
            # Set backend to 'brave' (verified to be robust and captcha-free)
            results = ddgs.text(query, max_results=10, backend="brave")
            output = []
            for r in results:
                # ddgs uses 'href' for the URL field
                url = r.get('href') or r.get('url') or 'No URL'
                output.append(f"Title: {r.get('title')}\nURL: {url}\n---")
            return "\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

def create_crew(user_query: str, step_callback_creator=None) -> Crew:
    # Initialize the LLM using Groq (defaulting to Llama 8b)
    groq_llm = LLM(
        model=os.environ.get("MODEL", "groq/llama3-8b-8192"),
        api_key=os.environ.get("GROQ_API_KEY"),
        additional_params={
            "parallel_tool_calls": False, # Prevent parallel tool hallucinations
            "num_retries": 5              # Auto-retry with backoff on rate limits
        }
    )

    # Scoped tool to programmatically enforce only one website scrape per execution
    scraped_urls = set()

    @tool("Scrape Website")
    def scrape_website(url: str) -> str:
        """Fetch the text content of a specific website URL to get more detail.
        You can ONLY call this tool ONCE. Select the single most relevant URL from your search results.
        Do NOT call this tool multiple times.
        """
        if len(scraped_urls) >= 1:
            return "Error: You are only allowed to scrape ONE website. You have already scraped a website. Please compile your report using the information you already gathered."
            
        try:
            # Strip trailing punctuation the LLM might append
            url = url.strip().rstrip(')').rstrip('.')
            with DDGS() as ddgs:
                res = ddgs.extract(url, fmt="text_markdown")
                scraped_urls.add(url)
                return res.get("content", "")[:3000]
        except Exception as e:
            return f"Scraping error: {str(e)}"

    # Get callbacks if provided
    researcher_callback = step_callback_creator("Senior Researcher") if step_callback_creator else None
    writer_callback = step_callback_creator("Technical Writer") if step_callback_creator else None

    # Define a researcher agent
    researcher = Agent(
        role="Senior Researcher",
        goal="Search the web and select the single most relevant URL for the topic.",
        backstory="You are an expert researcher. Your sole job is to use 'Web Search' to find information, evaluate the titles and URLs, select the SINGLE most promising URL to answer the query, and hand it off to the writer. You do not scrape.",
        verbose=True,
        llm=groq_llm,
        tools=[web_search],
        step_callback=researcher_callback
    )

    # Define a writer agent
    writer = Agent(
        role="Technical Writer",
        goal="Scrape the selected website and write a comprehensive, beautifully structured markdown report.",
        backstory="You are a professional technical writer and analyst. You take the URL provided by the researcher, use 'Scrape Website' to fetch its content, read it, and synthesize it into clear, well-structured, and readable markdown documentation.",
        verbose=True,
        llm=groq_llm,
        tools=[scrape_website],
        step_callback=writer_callback
    )

    # Define tasks
    research_task = Task(
        description=f"Find the single most relevant URL for the query: {user_query}. Use 'Web Search' to search the internet, look at the titles and URLs, select the single best URL, and output that URL.",
        expected_output="The single most relevant URL as a plain string.",
        agent=researcher,
    )

    write_task = Task(
        description=f"Take the URL provided by the Researcher. Use 'Scrape Website' to fetch the text content of that URL, read it, and compile it into a comprehensive, beautifully structured markdown report answering the user's original query: {user_query}.",
        expected_output="A final, polished markdown response answering the user's query.",
        agent=writer,
    )

    # Assemble the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew

