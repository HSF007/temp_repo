import asyncio
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agents import create_agents


load_dotenv()


class ResearchState(TypedDict, total=False):
    topic: str
    research_data: str
    summary: str
    notification: str


def create_workflow():
    """Build the sequential three-agent LangGraph workflow."""

    research_agent, summarization_agent, notification_agent = create_agents()

    def run_research(state: ResearchState):
        result = research_agent(state["topic"])
        return {"research_data": result}

    def run_summarization(state: ResearchState):
        summary = summarization_agent(state["research_data"])
        return {"summary": summary}

    def run_notification(state: ResearchState):
        result = notification_agent(state["summary"])
        return {"notification": result}

    graph = StateGraph(ResearchState)

    graph.add_node("research_agent", run_research)
    graph.add_node("summarization_agent", run_summarization)
    graph.add_node("notification_agent", run_notification)

    graph.add_edge(START, "research_agent")
    graph.add_edge("research_agent", "summarization_agent")
    graph.add_edge("summarization_agent", "notification_agent")
    graph.add_edge("notification_agent", END)

    return graph.compile()


def is_valid_topic(topic: str) -> bool:
    """Basic input validation before starting the workflow."""
    if not topic or len(topic.strip()) < 3 or len(topic) > 200:
        return False

    # Reject input containing no letters or digits.
    return any(character.isalnum() for character in topic)


async def main():
    print("       MULTI-AGENT AI ASSISTANT")
    print("========================================")

    topic = input("Enter a technology topic: ").strip()

    if not is_valid_topic(topic):
        print("The agent will not be able to do the task. Please enter a valid topic.")
        return

    workflow = create_workflow()

    try:
        result = await workflow.ainvoke({"topic": topic})

        print("\n========== RESEARCH RESULTS ==========")
        print(result.get("research_data", "No research data returned."))

        print("\n========== FIVE-POINT SUMMARY ==========")
        print(result.get("summary", "No summary generated."))

        print("\n========== NOTIFICATION ==========")
        print(result.get("notification", "No notification status returned."))

    except Exception as exc:
        print(f"\nThe workflow could not be completed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())