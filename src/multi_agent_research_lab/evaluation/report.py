"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown with analysis and failure modes."""
    lines = [
        "# Benchmark Report",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (0-10) | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")

    if len(metrics) >= 2:
        base, multi = metrics[0], metrics[1]
        speedup = base.latency_seconds / multi.latency_seconds if multi.latency_seconds > 0 else 0
        lines.extend([
            "",
            "## Analysis",
            "",
            f"- **Latency**: Multi-agent took {multi.latency_seconds:.2f}s vs baseline {base.latency_seconds:.2f}s ({speedup:.2f}x ratio)",
            f"- **Cost**: Multi-agent ${multi.estimated_cost_usd or 0:.4f} vs baseline ${base.estimated_cost_usd or 0:.4f}",
            f"- **Quality**: Multi-agent {multi.quality_score or 0:.1f}/10 vs baseline {base.quality_score or 0:.1f}/10",
            "",
            "Multi-agent workflows trade latency and cost for higher quality output through specialized agents (researcher, analyst, writer) that each focus on one aspect of the task.",
        ])

    lines.extend([
        "",
        "## Failure Modes and Mitigations",
        "",
        "| Failure Mode | Description | Mitigation |",
        "|---|---|---|",
        "| Rate Limiting | LLM provider returns 429 when quota exceeded | `tenacity` retry with exponential backoff in `LLMClient` |",
        "| Infinite Loop | Supervisor keeps routing without converging | `max_iterations` guardrail in config (default: 6) |",
        "| Timeout | Single agent takes too long to respond | `timeout_seconds` config + OpenAI SDK timeout |",
        "| Hallucination | LLM generates unsupported claims | Analyst agent flags weak evidence; Critic agent (optional) for fact-checking |",
        "| Missing Search Results | Search provider unavailable | Mock fallback in `SearchClient` when Tavily is unavailable |",
        "| State Corruption | Agent writes invalid data to shared state | Pydantic schema validation on `ResearchState` fields |",
        "",
    ])

    return "\n".join(lines) + "\n"
