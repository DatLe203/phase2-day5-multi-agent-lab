"""Writer agent skeleton."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        with trace_span("writer_agent"):
            llm_client = LLMClient()

            sources_ref = "\n".join(
                f"[{i+1}] {s.title} — {s.url}" for i, s in enumerate(state.sources)
            )
            response = llm_client.complete(
                system_prompt=f"You are a technical writer for {state.request.audience}. Synthesize a clear, well-structured response with citations referencing source numbers [1], [2], etc.",
                user_prompt=f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}\n\nAnalysis:\n{state.analysis_notes}\n\nSources:\n{sources_ref}",
            )
            state.final_answer = response.content
            state.agent_results.append(AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "cost_usd": response.cost_usd},
            ))
            state.add_trace_event("writer_done", {})
            logger.info("Writer completed final answer")
        return state
