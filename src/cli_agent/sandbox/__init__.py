"""
Sandbox Module - DeepAgents Sandbox Integration

Implements secure code execution environment from research:
- DeepAgents sandbox integration (NOT custom Docker)
- Transactional filesystem snapshots
- Git worktrees for parallel agent work
- Resource limits and isolation
"""

from .deepagents_sandbox import DeepAgentsSandbox, SandboxConfig, ExecutionResult
from .filesystem import TransactionalFS, Snapshot, ChangeSet
from .worktrees import WorktreeManager, Worktree

__all__ = [
    "DeepAgentsSandbox",
    "SandboxConfig",
    "ExecutionResult",
    "TransactionalFS",
    "Snapshot",
    "ChangeSet",
    "WorktreeManager",
    "Worktree",
]
