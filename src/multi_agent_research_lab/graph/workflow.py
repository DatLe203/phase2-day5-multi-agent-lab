"""LangGraph workflow skeleton."""

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    state: ResearchState


def _supervisor_node(data: dict[str, Any]) -> dict[str, Any]:
    data["state"] = SupervisorAgent().run(data["state"])
    return data


def _researcher_node(data: dict[str, Any]) -> dict[str, Any]:
    data["state"] = ResearcherAgent().run(data["state"])
    return data


def _analyst_node(data: dict[str, Any]) -> dict[str, Any]:
    data["state"] = AnalystAgent().run(data["state"])
    return data


def _writer_node(data: dict[str, Any]) -> dict[str, Any]:
    data["state"] = WriterAgent().run(data["state"])
    return data


def _route_after_supervisor(data: dict[str, Any]) -> str:
    last_route = data["state"].route_history[-1] if data["state"].route_history else "done"
    if last_route in ("researcher", "analyst", "writer"):
        return last_route
    return "done"


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> StateGraph:
        """Create a LangGraph graph."""
        graph = StateGraph(GraphState)

        graph.add_node("supervisor", _supervisor_node)
        graph.add_node("researcher", _researcher_node)
        graph.add_node("analyst", _analyst_node)
        graph.add_node("writer", _writer_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges("supervisor", _route_after_supervisor, {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "done": END,
        })
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        graph = self.build()
        compiled = graph.compile()
        result = compiled.invoke({"state": state})
        logger.info("Workflow completed in %d iterations", result["state"].iteration)
        return result["state"]
