"""
CLI Coding Agent - Autonomous Coding Agent with Advanced Architecture (2025-2026)

Implements the 7-layer architecture from research synthesis:
- Layer 1: Orchestrator (RLM-based) for unlimited context handling
- Layer 2: Specialist Agent Pool (Planner, Coder, Tester, Reviewer)
- Layer 3: Memory Infrastructure (SimpleMem, A-MEM, AgeMem)
- Layer 4: Code Understanding (Tree-sitter, LSP, SCIP, Graphiti)
- Layer 5: Execution Environment (DeepAgents Sandbox, Git Worktrees)
- Layer 6: Validation Pipeline (AgentCoder, Multi-Agent Reflexion)
- Layer 7: Observability (OpenTelemetry, Langfuse, Loop Detection)

Key Research Papers Implemented:
- arxiv:2512.24601 (RLM - Recursive Language Models)
- arxiv:2601.02553 (SimpleMem)
- arxiv:2601.01885 (AgeMem)
- NeurIPS 2025 (A-MEM - Zettelkasten)
- arxiv:2512.20845 (Multi-Agent Reflexion)
- arxiv:2510.00615 (ACON Framework)
- MetaGPT, OpenHands, LATS, AgentCoder patterns
"""

__version__ = "0.1.0"
__author__ = "CLI Coding Agent Team"

# Core
from .core.config import AgentConfig, load_config
from .core.state import AgentState, TaskState
from .core.rlm import RLMEngine

# Memory
from .memory.simple_mem import SimpleMem
from .memory.a_mem import AMem
from .memory.age_mem import AgeMem
from .memory.episodic import EpisodicMemory
from .memory.manager import MemoryManager

# Planning
from .planning.mcts import MCTSPlanner
from .planning.tree_of_thoughts import TreeOfThoughts
from .planning.loongflow import LoongFlowPES
from .planning.veriplan import VeriPlan
from .planning.multi_island import MultiIslandEvolution

# Agents
from .agents.mar import MultiAgentReflexion
from .agents.orchestrator import Orchestrator
from .agents.subagents import (
    PlannerAgent,
    CoderAgent,
    TesterAgent,
    ReviewerAgent,
    ResearcherAgent,
)
from .agents.agent_coder import AgentCoder

# Tools
from .tools.tool_registry import ToolRegistry, create_default_registry

# Sandbox
from .sandbox.deepagents_sandbox import DeepAgentsSandbox
from .sandbox.filesystem import TransactionalFS
from .sandbox.worktrees import WorktreeManager

# Observability
from .observability.tracing import Tracer
from .observability.metrics import MetricsCollector
from .observability.loop_detection import LoopDetector

# UI
from .ui.terminal import TerminalUI

# Main
from .main import CLIAgent, run

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentConfig",
    "load_config",
    "AgentState",
    "TaskState",
    "RLMEngine",
    # Memory
    "SimpleMem",
    "AMem",
    "AgeMem",
    "EpisodicMemory",
    "MemoryManager",
    # Planning
    "MCTSPlanner",
    "TreeOfThoughts",
    "LoongFlowPES",
    "VeriPlan",
    "MultiIslandEvolution",
    # Agents
    "MultiAgentReflexion",
    "Orchestrator",
    "PlannerAgent",
    "CoderAgent",
    "TesterAgent",
    "ReviewerAgent",
    "ResearcherAgent",
    "AgentCoder",
    # Tools
    "ToolRegistry",
    "create_default_registry",
    # Sandbox
    "DeepAgentsSandbox",
    "TransactionalFS",
    "WorktreeManager",
    # Observability
    "Tracer",
    "MetricsCollector",
    "LoopDetector",
    # UI
    "TerminalUI",
    # Main
    "CLIAgent",
    "run",
]
