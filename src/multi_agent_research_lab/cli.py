"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentResult, AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_traces
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()
    response = llm.complete(
        system_prompt="You are a helpful research assistant. Provide a comprehensive answer to the user's query.",
        user_prompt=query,
    )
    state.final_answer = response.content
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = "Research GraphRAG state-of-the-art and write a 500-word summary",
) -> None:
    """Run single-agent vs multi-agent benchmark and generate report."""
    _init()

    def baseline_runner(q: str) -> ResearchState:
        request = ResearchQuery(query=q)
        state = ResearchState(request=request)
        llm = LLMClient()
        response = llm.complete(
            system_prompt="You are a helpful research assistant. Provide a comprehensive answer.",
            user_prompt=q,
        )
        state.final_answer = response.content
        state.agent_results.append(AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "cost_usd": response.cost_usd},
        ))
        return state

    def multi_agent_runner(q: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(state)

    console.print("[bold]Running baseline benchmark...[/bold]")
    _, baseline_metrics = run_benchmark("single-agent-baseline", query, baseline_runner)
    console.print(f"  Baseline: {baseline_metrics.latency_seconds:.2f}s, cost=${baseline_metrics.estimated_cost_usd or 0:.4f}")

    console.print("[bold]Running multi-agent benchmark...[/bold]")
    _, multi_metrics = run_benchmark("multi-agent-workflow", query, multi_agent_runner)
    console.print(f"  Multi-agent: {multi_metrics.latency_seconds:.2f}s, cost=${multi_metrics.estimated_cost_usd or 0:.4f}")

    report = render_markdown_report([baseline_metrics, multi_metrics])

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")
    console.print(f"\n[green]Report saved to {report_path}[/green]")

    trace_path = export_traces()
    console.print(f"[green]Traces saved to {trace_path}[/green]")


if __name__ == "__main__":
    app()
