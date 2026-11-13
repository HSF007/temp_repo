import os

from langchain_openai import ChatOpenAI

from tools import web_search, save_summary


def create_agents():
    """Create the three agents used by the research workflow."""

    client = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
        temperature=0
    )

    # Agent 1: Research Agent
    def research_agent(topic: str) -> str:
        """Search the web for the user's technology topic."""
        return web_search.invoke({"topic": topic})

    # Agent 2: Summarization Agent
    def summarization_agent(research_data: str) -> str:
        """Summarize research data into five concise bullet points."""

        if research_data.startswith(("NO_RESULTS:", "SEARCH_ERROR:", "INVALID_INPUT:")):
            return (
                "Research summary unavailable.\n"
                f"- {research_data}"
            )

        prompt = f"""
You are a technology research summarization agent.

Read the web research below and produce exactly FIVE concise bullet points.
- Include only important information supported by the research.
- Make the points easy to understand.
- Avoid repeating the same information.
- Do not invent facts.
- If the research is insufficient or unclear, say so.

WEB RESEARCH:
{research_data}
"""
        response = client.invoke(prompt)
        return response.content if isinstance(response.content, str) else str(response.content)

    # Agent 3: Notification Agent
    def notification_agent(summary: str) -> str:
        """Save the summary in a text file."""
        return save_summary.invoke({"summary": summary})

    return research_agent, summarization_agent, notification_agent