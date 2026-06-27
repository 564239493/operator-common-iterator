"""ExecuterAgent sub-graph: 3-step test case execution pipeline.

Flow (mirrors the ExecuterAgent card in the front-end):
    exec_generate_atk → exec_cpu_derivation → exec_run_atk → END

Each internal node is wrapped with ``@traced_node`` here so spans + SSE events
fire under ``agent_id="execute"`` even when this sub-graph is nested inside the
main pipeline graph.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.executer_subgraph.generate_atk import exec_generate_atk_node
from agent.nodes.executer_subgraph.cpu_derivation import exec_cpu_derivation_node
from agent.nodes.executer_subgraph.run_atk import exec_run_atk_node
from agent.nodes.state import PipelineState
from agent.runtime import traced_node

__all__ = [
    "exec_generate_atk_node",
    "exec_cpu_derivation_node",
    "exec_run_atk_node",
    "create_executer_subgraph",
]


def _should_continue_after_generate(state: dict) -> str:
    if state.get("error"):
        return END
    return "exec_cpu_derivation"


def _should_continue_after_derivation(state: dict) -> str:
    if state.get("error"):
        return END
    return "exec_run_atk"


def create_executer_subgraph() -> CompiledStateGraph:
    """Build the 3-step ExecuterAgent sub-graph using :class:`PipelineState`."""
    graph = StateGraph(PipelineState)
    graph.add_node("exec_generate_atk", traced_node("exec_generate_atk")(exec_generate_atk_node))
    graph.add_node("exec_cpu_derivation", traced_node("exec_cpu_derivation")(exec_cpu_derivation_node))
    graph.add_node("exec_run_atk", traced_node("exec_run_atk")(exec_run_atk_node))

    graph.add_edge(START, "exec_generate_atk")
    graph.add_conditional_edges("exec_generate_atk", _should_continue_after_generate)
    graph.add_conditional_edges("exec_cpu_derivation", _should_continue_after_derivation)
    graph.add_edge("exec_run_atk", END)
    return graph.compile(name="executer-subgraph")
