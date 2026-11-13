from pathlib import Path

from ddgs import DDGS
from langchain_core.tools import tool


OUTPUT_FILE = Path(__file__).parent / "summary.txt"


@tool
def web_search(topic: str) -> str:
    """Search the web for technology-related information."""

    try:
        if not topic or len(topic.strip()) < 3:
            return "INVALID_INPUT: The agent will not be able to do the task."

        results = DDGS().text(
            topic,
            max_results=5
        )

        if not results:
            return "NO_RESULTS: No search results found."

        return "\n".join(
            f"Title: {item.get('title', '')}\n"
            f"Body: {item.get('body', '')}\n"
            f"URL: {item.get('href', '')}"
            for item in results
        )

    except Exception as e:
        return f"SEARCH_ERROR: {str(e)}"


@tool
def save_summary(summary: str) -> str:
    """Save the generated summary into summary.txt."""
    try:
        OUTPUT_FILE.write_text(summary.strip() + "\n", encoding="utf-8")
        return f"Summary successfully saved to: {OUTPUT_FILE}"
    except OSError as exc:
        return f"FILE_ERROR: Could not save the summary. Details: {exc}"