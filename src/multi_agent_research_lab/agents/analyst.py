"""Analyst agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        with trace_span("analyst_agent"):
            llm_client = LLMClient()
            response = llm_client.complete(
                system_prompt="You are a research analyst. Extract key claims, compare viewpoints, identify consensus and disagreements, and flag any weak or unsupported evidence.",
                user_prompt=f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}",
            )
            state.analysis_notes = response.content
            state.agent_results.append(AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "cost_usd": response.cost_usd},
            ))
            state.add_trace_event("analyst_done", {})
            logger.info("Analyst completed analysis")
        return state
