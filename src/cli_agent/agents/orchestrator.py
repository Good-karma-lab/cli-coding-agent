"""
Orchestrator Agent - Central Coordination for Multi-Agent System

Based on research for agent coordination:
- Central coordinator for task delegation
- Dynamic agent selection based on task requirements
- Message routing between specialized agents
- State management across agent interactions
- Hierarchical task decomposition
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable, Set
from enum import Enum
from datetime import datetime
import litellm
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """Types of messages between agents."""
    TASK = "task"
    QUERY = "query"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"
    RESULT = "result"
    FEEDBACK = "feedback"
    DELEGATION = "delegation"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""
    PLANNING = "planning"
    CODING = "coding"
    TESTING = "testing"
    REVIEWING = "reviewing"
    RESEARCHING = "researching"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTING = "documenting"
    DEPLOYING = "deploying"


@dataclass
class AgentMessage:
    """
    Message passed between agents.

    Supports typed messaging with metadata
    for routing and tracking.
    """
    id: str
    message_type: MessageType
    sender: str
    recipient: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    responded_at: Optional[datetime] = None
    parent_id: Optional[str] = None  # For message threading

    def create_response(
        self,
        content: Any,
        message_type: MessageType = MessageType.RESPONSE,
    ) -> "AgentMessage":
        """Create a response to this message."""
        return AgentMessage(
            id=str(uuid.uuid4()),
            message_type=message_type,
            sender=self.recipient,
            recipient=self.sender,
            content=content,
            parent_id=self.id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": str(self.content)[:100] if self.content else None,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
        }


@dataclass
class TaskAssignment:
    """
    Task assigned to an agent.

    Tracks task lifecycle from assignment
    to completion.
    """
    id: str
    task_description: str
    assigned_agent: str
    priority: TaskPriority = TaskPriority.MEDIUM
    capabilities_required: List[AgentCapability] = field(default_factory=list)

    # Task state
    status: str = "pending"  # pending, in_progress, completed, failed, blocked
    result: Optional[Any] = None
    error: Optional[str] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    blocked_by: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Subtasks
    subtasks: List["TaskAssignment"] = field(default_factory=list)
    parent_id: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.status in ("completed", "failed")

    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.task_description[:100],
            "agent": self.assigned_agent,
            "priority": self.priority.value,
            "status": self.status,
            "duration": self.duration,
            "num_subtasks": len(self.subtasks),
        }


class AgentRegistry:
    """Registry of available agents and their capabilities."""

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._capability_index: Dict[AgentCapability, Set[str]] = {
            cap: set() for cap in AgentCapability
        }

    def register(
        self,
        agent_id: str,
        agent_instance: Any,
        capabilities: List[AgentCapability],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an agent."""
        self._agents[agent_id] = {
            "instance": agent_instance,
            "capabilities": capabilities,
            "metadata": metadata or {},
            "registered_at": datetime.now(),
        }

        for cap in capabilities:
            self._capability_index[cap].add(agent_id)

    def unregister(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self._agents:
            caps = self._agents[agent_id]["capabilities"]
            for cap in caps:
                self._capability_index[cap].discard(agent_id)
            del self._agents[agent_id]

    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get agent instance by ID."""
        if agent_id in self._agents:
            return self._agents[agent_id]["instance"]
        return None

    def find_by_capability(self, capability: AgentCapability) -> List[str]:
        """Find agents with a specific capability."""
        return list(self._capability_index.get(capability, set()))

    def find_by_capabilities(
        self,
        capabilities: List[AgentCapability],
        require_all: bool = True,
    ) -> List[str]:
        """Find agents with specified capabilities."""
        if not capabilities:
            return list(self._agents.keys())

        matching = set(self._agents.keys())

        for cap in capabilities:
            agents_with_cap = self._capability_index.get(cap, set())
            if require_all:
                matching &= agents_with_cap
            else:
                matching |= agents_with_cap

        return list(matching)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [
            {
                "id": agent_id,
                "capabilities": [c.value for c in info["capabilities"]],
                "metadata": info["metadata"],
            }
            for agent_id, info in self._agents.items()
        ]


class Orchestrator:
    """
    Central orchestration agent for multi-agent coordination.

    Responsibilities:
    - Task decomposition and assignment
    - Agent selection based on capabilities
    - Message routing between agents
    - Progress tracking and error handling
    - State management across interactions
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_concurrent_tasks: int = 5,
        task_timeout: float = 300.0,  # 5 minutes
    ):
        self.model = model
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout

        self.registry = AgentRegistry()
        self._tasks: Dict[str, TaskAssignment] = {}
        self._message_history: List[AgentMessage] = []
        self._task_counter = 0

        # Event handlers
        self._on_task_complete: List[Callable] = []
        self._on_task_failed: List[Callable] = []

    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter}"

    def register_agent(
        self,
        agent_id: str,
        agent_instance: Any,
        capabilities: List[AgentCapability],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an agent with the orchestrator."""
        self.registry.register(agent_id, agent_instance, capabilities, metadata)

    async def process_request(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a user request.

        Decomposes request into tasks, assigns to agents,
        and coordinates execution.
        """
        # Analyze request and decompose into tasks
        tasks = await self._decompose_request(request, context)

        if not tasks:
            return {
                "status": "error",
                "message": "Could not decompose request into tasks",
            }

        # Execute tasks
        results = await self._execute_tasks(tasks, context)

        # Aggregate results
        final_result = await self._aggregate_results(request, results, context)

        return final_result

    async def _decompose_request(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> List[TaskAssignment]:
        """Decompose request into tasks with agent assignments."""
        # Get available agents
        available_agents = self.registry.list_agents()

        prompt = f"""Analyze this request and decompose it into tasks for a multi-agent system.

Request: {request}

Available agents and their capabilities:
{self._format_agents(available_agents)}

For each task, specify:
1. Task description
2. Which agent should handle it (by capability)
3. Priority: critical, high, medium, or low
4. Dependencies: which tasks must complete first (by task number)

Format each task as:
TASK <number>:
DESCRIPTION: <what needs to be done>
CAPABILITY: <required capability>
PRIORITY: <priority level>
DEPENDS_ON: <comma-separated task numbers, or "none">

Decompose the request into 1-5 focused tasks."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # Parse tasks
        tasks = self._parse_tasks(response.choices[0].message.content)

        # Assign agents to tasks
        for task in tasks:
            agent_id = self._select_agent(task.capabilities_required)
            if agent_id:
                task.assigned_agent = agent_id
            else:
                # Default to first available agent
                if available_agents:
                    task.assigned_agent = available_agents[0]["id"]

        return tasks

    async def _execute_tasks(
        self,
        tasks: List[TaskAssignment],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute tasks respecting dependencies."""
        # Store tasks
        for task in tasks:
            self._tasks[task.id] = task

        results: Dict[str, Any] = {}
        pending = list(tasks)
        running: Dict[str, asyncio.Task] = {}

        while pending or running:
            # Start ready tasks (dependencies met)
            ready_tasks = [
                t for t in pending
                if self._dependencies_met(t)
                and len(running) < self.max_concurrent_tasks
            ]

            for task in ready_tasks:
                pending.remove(task)
                task.status = "in_progress"
                task.started_at = datetime.now()

                # Create async task
                async_task = asyncio.create_task(
                    self._execute_single_task(task, context, results)
                )
                running[task.id] = async_task

            # Wait for any running task to complete
            if running:
                done, _ = await asyncio.wait(
                    running.values(),
                    timeout=self.task_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks
                for completed_task in done:
                    # Find task ID
                    task_id = None
                    for tid, atask in list(running.items()):
                        if atask == completed_task:
                            task_id = tid
                            del running[tid]
                            break

                    if task_id:
                        try:
                            result = completed_task.result()
                            results[task_id] = result
                            self._tasks[task_id].status = "completed"
                            self._tasks[task_id].result = result
                            self._tasks[task_id].completed_at = datetime.now()

                            # Notify handlers
                            for handler in self._on_task_complete:
                                handler(self._tasks[task_id])

                        except Exception as e:
                            results[task_id] = {"error": str(e)}
                            self._tasks[task_id].status = "failed"
                            self._tasks[task_id].error = str(e)
                            self._tasks[task_id].completed_at = datetime.now()

                            # Notify handlers
                            for handler in self._on_task_failed:
                                handler(self._tasks[task_id], e)

        return results

    async def _execute_single_task(
        self,
        task: TaskAssignment,
        context: Dict[str, Any],
        completed_results: Dict[str, Any],
    ) -> Any:
        """Execute a single task."""
        agent = self.registry.get_agent(task.assigned_agent)

        if agent is None:
            raise ValueError(f"Agent {task.assigned_agent} not found")

        # Prepare task context with results from dependencies
        task_context = context.copy()
        for dep_id in task.depends_on:
            if dep_id in completed_results:
                task_context[f"result_{dep_id}"] = completed_results[dep_id]

        # Send task message to agent
        message = AgentMessage(
            id=str(uuid.uuid4()),
            message_type=MessageType.TASK,
            sender="orchestrator",
            recipient=task.assigned_agent,
            content={
                "task_id": task.id,
                "description": task.task_description,
                "context": task_context,
                "priority": task.priority.value,
            },
        )

        self._message_history.append(message)

        # Execute via agent
        if hasattr(agent, "execute"):
            result = await agent.execute(task.task_description, task_context)
        elif hasattr(agent, "run"):
            result = await agent.run(task.task_description, task_context)
        else:
            # Fallback: use LLM directly
            result = await self._execute_with_llm(task, task_context)

        # Record response
        response = message.create_response(result, MessageType.RESULT)
        self._message_history.append(response)

        return result

    async def _execute_with_llm(
        self,
        task: TaskAssignment,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task directly with LLM when no agent available."""
        prompt = f"""Execute this task:

Task: {task.task_description}

Context:
{self._format_context(context)}

Provide a complete response."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        return {
            "content": response.choices[0].message.content,
            "executed_by": "llm_fallback",
        }

    async def _aggregate_results(
        self,
        request: str,
        results: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate results from all tasks."""
        # Format results for aggregation
        results_text = []
        for task_id, result in results.items():
            task = self._tasks.get(task_id)
            task_desc = task.task_description if task else task_id
            results_text.append(f"Task: {task_desc}\nResult: {result}")

        prompt = f"""Aggregate these task results into a final response.

Original request: {request}

Task results:
{chr(10).join(results_text)}

Provide:
1. A summary of what was accomplished
2. The final result or answer
3. Any issues or follow-up items

Format:
SUMMARY: <brief summary>
RESULT: <the main result>
ISSUES: <any issues, or "none">"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = response.choices[0].message.content

        # Parse response
        summary = ""
        result = ""
        issues = []

        for line in text.split("\n"):
            if line.upper().startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()
            elif line.upper().startswith("RESULT:"):
                result = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ISSUES:"):
                issues_text = line.split(":", 1)[1].strip()
                if issues_text.lower() != "none":
                    issues = [issues_text]

        return {
            "status": "success",
            "summary": summary,
            "result": result or text,
            "issues": issues,
            "task_results": results,
            "tasks_completed": sum(
                1 for t in self._tasks.values() if t.status == "completed"
            ),
            "tasks_failed": sum(
                1 for t in self._tasks.values() if t.status == "failed"
            ),
        }

    def _select_agent(
        self,
        capabilities: List[AgentCapability],
    ) -> Optional[str]:
        """Select the best agent for required capabilities."""
        # Find agents that have all required capabilities
        matching = self.registry.find_by_capabilities(capabilities, require_all=True)

        if matching:
            return matching[0]

        # Fall back to agents with any of the capabilities
        matching = self.registry.find_by_capabilities(capabilities, require_all=False)

        if matching:
            return matching[0]

        return None

    def _dependencies_met(self, task: TaskAssignment) -> bool:
        """Check if task dependencies are met."""
        for dep_id in task.depends_on:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                return False
        return True

    def _parse_tasks(self, text: str) -> List[TaskAssignment]:
        """Parse tasks from LLM response."""
        tasks = []
        current_task = None
        task_id_map: Dict[int, str] = {}

        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            upper = line.upper()

            if upper.startswith("TASK"):
                # Save previous task
                if current_task:
                    tasks.append(current_task)

                # Start new task
                task_id = self._generate_task_id()
                task_num = len(tasks) + 1
                task_id_map[task_num] = task_id

                current_task = TaskAssignment(
                    id=task_id,
                    task_description="",
                    assigned_agent="",
                )

            elif current_task:
                if upper.startswith("DESCRIPTION:"):
                    current_task.task_description = line.split(":", 1)[1].strip()

                elif upper.startswith("CAPABILITY:"):
                    cap_str = line.split(":", 1)[1].strip().lower()
                    try:
                        cap = AgentCapability(cap_str)
                        current_task.capabilities_required.append(cap)
                    except ValueError:
                        # Try to match partial
                        for c in AgentCapability:
                            if c.value in cap_str or cap_str in c.value:
                                current_task.capabilities_required.append(c)
                                break

                elif upper.startswith("PRIORITY:"):
                    priority_str = line.split(":", 1)[1].strip().lower()
                    try:
                        current_task.priority = TaskPriority(priority_str)
                    except ValueError:
                        pass

                elif upper.startswith("DEPENDS_ON:"):
                    deps_str = line.split(":", 1)[1].strip().lower()
                    if deps_str != "none":
                        import re
                        dep_nums = re.findall(r"\d+", deps_str)
                        for num_str in dep_nums:
                            num = int(num_str)
                            if num in task_id_map:
                                current_task.depends_on.append(task_id_map[num])

        # Add last task
        if current_task:
            tasks.append(current_task)

        return tasks

    def _format_agents(self, agents: List[Dict[str, Any]]) -> str:
        """Format agent list for prompts."""
        lines = []
        for agent in agents:
            caps = ", ".join(agent["capabilities"])
            lines.append(f"- {agent['id']}: {caps}")
        return "\n".join(lines) if lines else "(no agents registered)"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompts."""
        lines = []
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 100:
                lines.append(f"  {key}: {value[:100]}...")
            elif isinstance(value, (list, dict)):
                lines.append(f"  {key}: {type(value).__name__} ({len(value)} items)")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def on_task_complete(self, handler: Callable) -> None:
        """Register task completion handler."""
        self._on_task_complete.append(handler)

    def on_task_failed(self, handler: Callable) -> None:
        """Register task failure handler."""
        self._on_task_failed.append(handler)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks."""
        return [task.to_dict() for task in self._tasks.values()]

    def get_message_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent message history."""
        return [m.to_dict() for m in self._message_history[-limit:]]

    def clear_state(self) -> None:
        """Clear orchestrator state."""
        self._tasks.clear()
        self._message_history.clear()
        self._task_counter = 0
