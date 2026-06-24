"""Researcher agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        with trace_span("researcher_agent") as span:
            search_client = SearchClient()
            llm_client = LLMClient()

            sources = search_client.search(state.request.query, state.request.max_sources)
            state.sources.extend(sources)

            sources_text = "\n".join(
                f"[{i+1}] {s.title}: {s.snippet}" for i, s in enumerate(sources)
            )
            response = llm_client.complete(
                system_prompt="You are a research assistant. Summarize the following sources into concise research notes. Include key findings and cite sources by number.",
                user_prompt=f"Query: {state.request.query}\n\nSources:\n{sources_text}",
            )
            state.research_notes = response.content
            state.agent_results.append(AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "cost_usd": response.cost_usd},
            ))
            state.add_trace_event("researcher_done", {"sources_count": len(sources)})
            span["attributes"]["sources_count"] = len(sources)
            logger.info("Researcher found %d sources", len(sources))
        return state
