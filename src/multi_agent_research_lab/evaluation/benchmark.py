"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality and return metrics."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    total_cost = sum(
        r.metadata.get("cost_usd", 0) or 0 for r in state.agent_results
    )

    quality = _score_quality(state)

    notes_parts = [f"iterations={state.iteration}", f"sources={len(state.sources)}"]
    if state.errors:
        notes_parts.append(f"errors={len(state.errors)}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost if total_cost > 0 else None,
        quality_score=quality,
        notes=", ".join(notes_parts),
    )
    return state, metrics


def _score_quality(state: ResearchState) -> float:
    """Simple heuristic quality score (0-10)."""
    score = 0.0
    if state.final_answer:
        score += 3.0
        if len(state.final_answer) > 200:
            score += 2.0
        if any(f"[{i}]" in state.final_answer for i in range(1, 6)):
            score += 2.0
    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.0
    if state.sources:
        score += 1.0
    return min(score, 10.0)
