"""Build a deep research agent and record its LLM and tool calls for analysis."""

__all__ = ["ResearchAgent"]


def __getattr__(name):
    # Keep corpus and citation helpers usable without importing LangGraph.
    if name == "ResearchAgent":
        from .api import ResearchAgent

        return ResearchAgent
    raise AttributeError(name)
