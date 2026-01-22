"""
Agent State Management.

Implements state structures for:
- AgentState: Global orchestrator state
- TaskState: Individual task execution state
- PlanNode: MCTS/ToT planning tree nodes
- MemoryUnit: SimpleMem memory units
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    """MCTS/ToT tree node status."""
    UNEXPLORED = "unexplored"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    SELECTED = "selected"


class MemoryUnitType(str, Enum):
    """Types of memory units in SimpleMem."""
    ATOMIC_FACT = "atomic_fact"  # Single fact with coreference resolved
    EPISODE = "episode"  # Sequence of actions
    INSIGHT = "insight"  # Consolidated abstract knowledge
    LESSON = "lesson"  # Learned from failure (Reflexion)
    DECISION = "decision"  # Architectural/design decision
    PREFERENCE = "preference"  # User preference


# =============================================================================
# Pydantic Models for Structured Output (reduces hallucinations)
# =============================================================================

class CodeChange(BaseModel):
    """Structured code change output - enforced via Pydantic."""
    file_path: str = Field(description="Path to the file to modify")
    operation: str = Field(description="create, modify, or delete")
    content: str = Field(description="New content for the file")
    rationale: str = Field(description="Why this change is needed")
    tests_required: list[str] = Field(default=[], description="Tests that should pass after this change")


class TestCase(BaseModel):
    """Structured test case from Test Designer Agent."""
    name: str = Field(description="Test function name")
    description: str = Field(description="What this test verifies")
    test_code: str = Field(description="The actual test code")
    expected_behavior: str = Field(description="Expected outcome")
    edge_cases: list[str] = Field(default=[], description="Edge cases covered")


class PlanStep(BaseModel):
    """Structured planning step."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = Field(description="What this step accomplishes")
    dependencies: list[str] = Field(default=[], description="Step IDs this depends on")
    agent_role: str = Field(default="coder", description="Which agent handles this")
    estimated_complexity: str = Field(default="medium", description="low, medium, high")
    verification_criteria: list[str] = Field(default=[], description="How to verify completion")
    rollback_strategy: Optional[str] = Field(default=None, description="How to undo if needed")


class ReflexionFeedback(BaseModel):
    """Feedback from Multi-Agent Reflexion."""
    attempt_number: int
    success: bool
    error_message: Optional[str] = None
    diagnosis: Optional[str] = Field(default=None, description="What went wrong")
    critique: Optional[str] = Field(default=None, description="Critical analysis")
    suggestion: Optional[str] = Field(default=None, description="How to improve")
    lesson_learned: Optional[str] = Field(default=None, description="For long-term memory")


# =============================================================================
# Memory Units (SimpleMem, A-MEM)
# =============================================================================

class MemoryUnit(BaseModel):
    """
    Memory unit for SimpleMem and A-MEM.

    Implements Semantic Lossless Compression principles:
    - Self-contained (no pronouns, absolute timestamps)
    - Multi-view indexed (embeddings + keywords + metadata)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryUnitType = MemoryUnitType.ATOMIC_FACT
    content: str = Field(description="The actual memory content")

    # Coreference resolution - all entities explicit
    entities: list[str] = Field(default=[], description="Named entities in this memory")

    # Temporal anchoring - absolute timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: Optional[datetime] = Field(default=None)
    valid_until: Optional[datetime] = Field(default=None)

    # Multi-view indexing
    keywords: list[str] = Field(default=[])
    embedding: Optional[list[float]] = Field(default=None)

    # A-MEM Zettelkasten linking
    linked_memories: list[str] = Field(default=[], description="IDs of related memories")
    link_strengths: dict[str, float] = Field(default={}, description="Semantic similarity scores")

    # Consolidation tracking
    abstraction_level: int = Field(default=0, description="0=atomic, higher=more abstract")
    source_memories: list[str] = Field(default=[], description="IDs of memories this consolidates")

    # Metadata
    source: str = Field(default="agent", description="Where this memory came from")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = Field(default=None)

    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()


# =============================================================================
# Planning Tree Nodes (MCTS/LATS/ToT)
# =============================================================================

class PlanNode(BaseModel):
    """
    Planning tree node for MCTS/LATS/Tree-of-Thoughts.

    Supports:
    - UCB selection with exploration bonus
    - Value estimation from LLM or execution feedback
    - Backpropagation of results
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = Field(default=None)
    children_ids: list[str] = Field(default=[])

    # Node content
    thought: str = Field(description="The reasoning/plan at this node")
    action: Optional[str] = Field(default=None, description="Action to take")
    observation: Optional[str] = Field(default=None, description="Result of action")

    # MCTS statistics
    visits: int = Field(default=0)
    total_value: float = Field(default=0.0)
    status: NodeStatus = NodeStatus.UNEXPLORED

    # Multi-Island tracking
    island_id: Optional[str] = Field(default=None, description="Which island this belongs to")

    # Verification
    verification_result: Optional[bool] = Field(default=None)
    verification_feedback: Optional[str] = Field(default=None)

    # MAP-Elites archive - even failed nodes can have useful parts
    useful_fragments: list[str] = Field(default=[], description="Reusable code/ideas from this node")

    depth: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def average_value(self) -> float:
        """Average value for UCB calculation."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, parent_visits: int, exploration_weight: float = 1.414) -> float:
        """Calculate UCB1 score for node selection."""
        import math
        if self.visits == 0:
            return float('inf')  # Unexplored nodes have infinite priority

        exploitation = self.average_value
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def backpropagate(self, value: float) -> None:
        """Update statistics after simulation."""
        self.visits += 1
        self.total_value += value


# =============================================================================
# Task State
# =============================================================================

class TaskState(BaseModel):
    """
    State for an individual task being executed.

    Tracks the full lifecycle from planning through validation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(description="What this task should accomplish")
    status: TaskStatus = TaskStatus.PENDING

    # Assignment
    assigned_agent: Optional[str] = Field(default=None, description="Which agent is handling this")
    parent_task_id: Optional[str] = Field(default=None, description="For subtask hierarchy")
    subtask_ids: list[str] = Field(default=[])

    # Planning (LoongFlow PES)
    plan: Optional[list[PlanStep]] = Field(default=None)
    plan_tree_root: Optional[str] = Field(default=None, description="Root PlanNode ID for MCTS")
    current_step_index: int = Field(default=0)

    # Execution
    git_branch: Optional[str] = Field(default=None, description="Git worktree branch for this task")
    worktree_path: Optional[str] = Field(default=None)
    changes: list[CodeChange] = Field(default=[])

    # AgentCoder validation
    generated_tests: list[TestCase] = Field(default=[])
    test_results: dict[str, bool] = Field(default={})

    # Multi-Agent Reflexion
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=5)
    reflexion_feedback: list[ReflexionFeedback] = Field(default=[])

    # Context (what the subagent actually sees)
    context_files: list[str] = Field(default=[], description="Files loaded into context")
    context_tokens: int = Field(default=0)

    # Results
    success: Optional[bool] = Field(default=None)
    error: Optional[str] = Field(default=None)
    output: Optional[str] = Field(default=None)
    lessons_learned: list[str] = Field(default=[], description="For SimpleMem")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Cost tracking
    tokens_used: int = Field(default=0)
    cost_usd: float = Field(default=0.0)

    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# =============================================================================
# Agent State (Orchestrator-level)
# =============================================================================

class AgentState(BaseModel):
    """
    Global agent state managed by the Orchestrator.

    Implements RLM principle: only task objectives, progress state,
    and artifact references in working memory. Substantial work
    delegated to subagents with clean contexts.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Mission context (what we're trying to accomplish)
    mission: Optional[str] = Field(default=None, description="High-level objective")
    constraints: list[str] = Field(default=[], description="Constraints on the solution")

    # Project understanding
    project_root: Optional[str] = Field(default=None)
    project_type: Optional[str] = Field(default=None, description="Detected project type")
    project_languages: list[str] = Field(default=[])

    # Code Understanding Graph references (not the full graph!)
    code_graph_indexed: bool = Field(default=False)
    code_graph_last_updated: Optional[datetime] = Field(default=None)
    key_files: list[str] = Field(default=[], description="Important files for current mission")
    key_symbols: list[str] = Field(default=[], description="Important symbols/functions")

    # Task management
    tasks: dict[str, TaskState] = Field(default={}, description="Task ID -> TaskState")
    task_order: list[str] = Field(default=[], description="Ordered list of task IDs")
    current_task_id: Optional[str] = Field(default=None)

    # Multi-Island state
    active_islands: dict[str, str] = Field(default={}, description="Island ID -> approach description")
    map_elites_archive: list[dict[str, Any]] = Field(default=[], description="Useful fragments from all attempts")

    # Memory references (not full memory, just IDs)
    relevant_memory_ids: list[str] = Field(default=[], description="Memory IDs relevant to current mission")
    session_insights: list[str] = Field(default=[], description="Insights from this session")

    # Progress tracking
    completed_tasks: int = Field(default=0)
    failed_tasks: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)

    # Loop detection
    recent_actions: list[str] = Field(default=[], description="Recent actions for loop detection")
    consecutive_failures: int = Field(default=0)

    # Session timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

    def add_task(self, task: TaskState) -> None:
        """Add a new task to the state."""
        self.tasks[task.id] = task
        self.task_order.append(task.id)

    def get_current_task(self) -> Optional[TaskState]:
        """Get the currently active task."""
        if self.current_task_id and self.current_task_id in self.tasks:
            return self.tasks[self.current_task_id]
        return None

    def record_action(self, action: str, max_history: int = 10) -> None:
        """Record an action for loop detection."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > max_history:
            self.recent_actions = self.recent_actions[-max_history:]
        self.last_activity = datetime.utcnow()

    def check_for_loops(self, similarity_threshold: float = 0.9) -> bool:
        """Check if recent actions indicate a loop."""
        if len(self.recent_actions) < 4:
            return False

        # Simple check: exact repeated action
        last_action = self.recent_actions[-1]
        repeat_count = sum(1 for a in self.recent_actions[-5:] if a == last_action)

        return repeat_count >= 3

    def update_costs(self, tokens: int, cost: float) -> None:
        """Update token and cost tracking."""
        self.total_tokens += tokens
        self.total_cost += cost

        if self.current_task_id and self.current_task_id in self.tasks:
            self.tasks[self.current_task_id].tokens_used += tokens
            self.tasks[self.current_task_id].cost_usd += cost


# =============================================================================
# RLM Context Environment
# =============================================================================

class RLMContext(BaseModel):
    """
    RLM (Recursive Language Model) context environment.

    Context stored as external object that LLM manipulates via code.
    Enables handling inputs 2 orders of magnitude beyond context windows.
    """
    # Full context stored externally
    files: dict[str, str] = Field(default={}, description="File path -> content")
    history: list[dict[str, str]] = Field(default=[], description="Conversation history")
    tool_outputs: list[dict[str, Any]] = Field(default=[], description="Previous tool results")

    # Metadata
    total_chars: int = Field(default=0)
    total_tokens_estimate: int = Field(default=0)

    # Access methods (these would be available to LLM via REPL)
    def search_files(self, pattern: str) -> list[str]:
        """Search file contents for pattern."""
        import re
        results = []
        for path, content in self.files.items():
            if re.search(pattern, content, re.IGNORECASE):
                results.append(path)
        return results

    def get_file_chunk(self, path: str, start_line: int, end_line: int) -> str:
        """Get specific lines from a file."""
        if path not in self.files:
            return ""
        lines = self.files[path].split('\n')
        return '\n'.join(lines[start_line:end_line])

    def filter_history(self, keyword: str) -> list[dict[str, str]]:
        """Filter history for relevant entries."""
        return [h for h in self.history if keyword.lower() in str(h).lower()]

    def get_recent_errors(self, n: int = 5) -> list[str]:
        """Get recent error messages from tool outputs."""
        errors = []
        for output in reversed(self.tool_outputs):
            if 'error' in str(output).lower():
                errors.append(str(output))
                if len(errors) >= n:
                    break
        return errors

    def summarize_file(self, path: str, max_lines: int = 50) -> str:
        """Get a summary of a file (first N lines + structure)."""
        if path not in self.files:
            return ""
        content = self.files[path]
        lines = content.split('\n')

        if len(lines) <= max_lines:
            return content

        # Return first lines + "..." + last few lines
        return '\n'.join(lines[:max_lines-5]) + f'\n... ({len(lines) - max_lines} more lines) ...\n' + '\n'.join(lines[-5:])
