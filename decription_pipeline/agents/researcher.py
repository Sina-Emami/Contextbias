from crewai import Agent, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool


def build_fact_checker_agent() -> Agent:
    # Use gpt-5-mini for research as requested
    research_llm = LLM(model="gpt-5-mini")
    return Agent(
        name="Reliable Fact Checker",
        role="Find factual, up-to-date, verifiable answers with citations.",
        goal="Answer each question with a single percentage, one-word category, or YES/NO; always cite a reputable source.",
        backstory="Uses search + scraping to retrieve trustworthy sources. Never guesses.",
        tools=[SerperDevTool(), ScrapeWebsiteTool()],
        llm=research_llm,
        allow_delegation=False,
        verbose=True,
    )
