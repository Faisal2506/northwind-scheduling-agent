"""LangGraph workflow: the agent node calls Gemini, the tool node calls the clinic."""

from agent.runtime import build_graph

__all__ = ["build_graph"]
