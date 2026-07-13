import time
from pydantic import BaseModel
from crewai import Agent
from crewai.flow.flow import Flow, listen, start
from ddgs import DDGS

from .utils import clean_markdown, extract_relevant_content
from .llm_config import get_groq_llm

# Workaround for CrewAI injecting cache_breakpoint on unsupported models
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda message: message
except ImportError:
    pass


class ResearchState(BaseModel):
    query: str = ""
    urls: list[str] = []
    scraped_data: list[str] = []
    report: str = ""


class ResearchFlow(Flow[ResearchState]):
    step_callback_creator = None
    status_callback = None

    def log_status(self, message: str):
        if self.status_callback:
            self.status_callback(message)
        print(f"[Flow Status] {message}")

    def extract_search_query(self) -> str:
        self.log_status("Formulating search query...")
        groq_llm = get_groq_llm(model="groq/llama-3.1-8b-instant")
        prompt = (
            f"You are a search query optimizer. Extract the core search keywords from this user request. "
            f"Output ONLY the optimized search query string (maximum 10 words), with no quotes or preamble.\n\n"
            f"User Request: {self.state.query}"
        )
        try:
            res = groq_llm.call(prompt)
            clean_q = res.strip().replace('"', '').replace("'", "")
            return clean_q
        except Exception as e:
            self.log_status(f"Failed to optimize query: {e}. Using raw query.")
            return self.state.query[:100]

    @start()
    def gather_urls(self):
        import json
        search_query = self.extract_search_query()
        self.log_status(f"Searching the web for: '{search_query}'")
        results = []
        try:
            with DDGS() as ddgs:
                try:
                    # 1. Increase Search Yield
                    results = list(ddgs.text(search_query, max_results=20, backend="auto"))
                except Exception as e:
                    self.log_status(f"DDGS search with auto backend failed: {e}")
                
                if not results:
                    self.state.urls = []
                    self.log_status("No URLs found from search.")
                    return

                # 2. Data Formatting
                formatted_results = []
                for i, r in enumerate(results):
                    url = r.get('href') or r.get('url')
                    title = r.get('title', 'No Title')
                    snippet = r.get('body', 'No Snippet')
                    formatted_results.append(f"Index: {i}\nTitle: {title}\nURL: {url}\nSnippet: {snippet}")
                
                results_str = "\n\n".join(formatted_results)
                self.log_status(f"Fetched {len(results)} results, curating top 5 with LLM...")

                # 3. LLM Curation (8B Model)
                groq_llm = get_groq_llm(model="groq/llama-3.1-8b-instant")
                prompt = (
                    f"You are an expert researcher evaluating search results for a user's query.\n"
                    f"User's Original Query: {self.state.query}\n\n"
                    f"Here are the top search results:\n\n{results_str}\n\n"
                    f"Select exactly the 5 most relevant and trustworthy results to answer the user's query."
                )

                class URLCuration(BaseModel):
                    selected_indices: list[int]

                try:
                    res = groq_llm.call(prompt, response_model=URLCuration)
                    # 4. State Update
                    import json
                    data = json.loads(res)
                    indices = data.get("selected_indices", [])
                    
                    selected_urls = []
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(results):
                            r = results[idx]
                            url = r.get('href') or r.get('url')
                            if url:
                                selected_urls.append(url)
                    
                    self.state.urls = selected_urls[:5]
                except Exception as e:
                    self.log_status(f"LLM curation failed: {e}. Falling back to top 5.")
                    self.state.urls = [r.get('href') or r.get('url') for r in results if r.get('href') or r.get('url')][:5]

                self.log_status(f"Curated {len(self.state.urls)} URLs to scrape.")
                print(f"\n=== CURATED URLS ===\n" + "\n".join(str(url) for url in self.state.urls if url) + "\n====================\n")
        except Exception as e:
            self.state.urls = []
            self.log_status(f"Search error: {e}")

    @listen(gather_urls)
    def scrape_websites(self):
        self.state.scraped_data = []
        for url in self.state.urls:
            self.log_status(f"Scraping URL: {url}")
            try:
                # Strip trailing punctuation the LLM might append (though here we got it programmatically)
                url = url.strip().rstrip(')').rstrip('.')
                with DDGS() as ddgs:
                    res = ddgs.extract(url, fmt="text_markdown")
                    content = res.get("content") or ""
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="ignore")
                    elif not isinstance(content, str):
                        content = str(content)
                    
                    # Instead of truncating just at the start, extract relevant paragraphs
                    if content:
                        cleaned_content = clean_markdown(content)
                        truncated = extract_relevant_content(cleaned_content, self.state.query, max_tokens=1500)
                        scraped_entry = f"URL: {url}\nContent Snippet:\n{truncated}\n---"
                        self.state.scraped_data.append(scraped_entry)
                        print(f"\n=== SCRAPED DATA FROM {url} ===\n{truncated}\n================================\n")
                    
                    # Add a small delay to avoid DDGS rate limits
                    time.sleep(1.5)
            except Exception as e:
                self.log_status(f"Scrape error for {url}: {e}")

    @listen(scrape_websites)
    def synthesize_report(self):
        self.log_status("Synthesizing final report...")
        groq_llm = get_groq_llm(
            model="groq/llama-3.3-70b-versatile", # Comment this out while experimenting to save tokens
            additional_params={
                "parallel_tool_calls": False,
                "num_retries": 5
            }
        )

        writer_callback = self.step_callback_creator("Technical Writer") if self.step_callback_creator else None

        writer = Agent(
            role="Technical Writer",
            goal="Synthesize the provided research data into a comprehensive, beautifully structured markdown report.",
            backstory="You are a professional technical writer and analyst. You synthesize raw research into clear, well-structured, and readable markdown documentation.",
            verbose=True,
            llm=groq_llm,
            max_rpm=10, 
            respect_context_window=True,
            step_callback=writer_callback
        )

        combined_data = "\n\n".join(self.state.scraped_data)
        if not combined_data:
            combined_data = "No data could be retrieved from the web."

        prompt = (
            f"Please write a comprehensive, beautifully structured markdown report answering the user's original query: '{self.state.query}'.\n\n"
            f"Here is the research data gathered from multiple websites:\n\n{combined_data}"
        )

        result = writer.kickoff(prompt)
        self.state.report = getattr(result, "raw", str(result))
        return self.state.report

def create_flow(user_query: str, step_callback_creator=None, status_callback=None) -> ResearchFlow:
    flow = ResearchFlow(name="ResearchFlow")
    flow.state.query = user_query
    flow.step_callback_creator = step_callback_creator
    flow.status_callback = status_callback
    return flow
