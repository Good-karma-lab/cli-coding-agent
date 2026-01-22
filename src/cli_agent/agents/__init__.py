"""
Agents Module - Multi-Agent Reflexion, Orchestrator, SubAgents, AgentCoder

Implements sophisticated multi-agent patterns from research:
- Multi-Agent Reflexion (MAR): Diverse reasoning personas
- Orchestrator: Central coordination agent
- SubAgents: Specialized agents (Planner, Coder, Tester, Reviewer)
- AgentCoder: Three-agent validation pattern
"""

from .mar import MultiAgentReflexion, ReflexionPersona, ReflexionResult
from .orchestrator import Orchestrator, AgentMessage, TaskAssignment
from .subagents import (
    BaseSubAgent,
    PlannerAgent,
    CoderAgent,
    TesterAgent,
    ReviewerAgent,
    ResearcherAgent,
)
from .agent_coder import AgentCoder, ProgrammerAgent, TestDesignerAgent, TestExecutorAgent

__all__ = [
    # MAR
    "MultiAgentReflexion",
    "ReflexionPersona",
    "ReflexionResult",
    # Orchestrator
    "Orchestrator",
    "AgentMessage",
    "TaskAssignment",
    # SubAgents
    "BaseSubAgent",
    "PlannerAgent",
    "CoderAgent",
    "TesterAgent",
    "ReviewerAgent",
    "ResearcherAgent",
    # AgentCoder
    "AgentCoder",
    "ProgrammerAgent",
    "TestDesignerAgent",
    "TestExecutorAgent",
]
