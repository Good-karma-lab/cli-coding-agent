"""
MCTS/LATS Planner - Language Agent Tree Search

Based on arxiv research for planning in coding agents:
- Monte Carlo Tree Search adapted for language agents
- UCB1 selection with exploration/exploitation balance
- Simulation via LLM rollouts
- Backpropagation of rewards through action tree
- Integration with reflexion for learning from failures
"""

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
import litellm
from pydantic import BaseModel


class ActionType(str, Enum):
    """Types of actions in the search tree."""
    THINK = "think"           # Reasoning step
    CODE = "code"             # Write/modify code
    TEST = "test"             # Run tests
    SEARCH = "search"         # Search codebase
    READ = "read"             # Read file
    EXECUTE = "execute"       # Execute command
    VERIFY = "verify"         # Verify solution
    REFINE = "refine"         # Refine approach


@dataclass
class MCTSNode:
    """
    Node in the MCTS search tree.

    Each node represents a state in the problem-solving process,
    with UCB1 scoring for selection during tree traversal.
    """
    id: str
    action: str
    action_type: ActionType
    state: Dict[str, Any]
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0

    # Reflexion integration
    reflections: List[str] = field(default_factory=list)
    failed_attempts: int = 0

    # Metadata
    depth: int = 0
    is_terminal: bool = False
    terminal_reward: Optional[float] = None

    @property
    def ucb1_score(self) -> float:
        """
        Calculate UCB1 score for node selection.

        UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))

        Where:
        - Q(s,a) = average reward for this action from this state
        - c = exploration constant (typically sqrt(2))
        - N(s) = visits to parent state
        - N(s,a) = visits to this state-action pair
        """
        if self.visits == 0:
            return float('inf')  # Unexplored nodes have infinite priority

        exploitation = self.total_reward / self.visits

        if self.parent is None or self.parent.visits == 0:
            exploration = 0
        else:
            exploration = math.sqrt(2) * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )

        return exploitation + exploration

    @property
    def average_reward(self) -> float:
        """Get average reward for this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def add_child(self, child: "MCTSNode") -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def add_reflection(self, reflection: str) -> None:
        """Add a reflection from failed attempt."""
        self.reflections.append(reflection)
        self.failed_attempts += 1

    def get_path(self) -> List["MCTSNode"]:
        """Get path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "action": self.action,
            "action_type": self.action_type.value,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "ucb1_score": self.ucb1_score,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "num_children": len(self.children),
            "reflections": self.reflections,
        }


class SimulationResult(BaseModel):
    """Result of a simulation rollout."""
    success: bool
    reward: float
    trajectory: List[str]
    final_state: Dict[str, Any]
    reflection: Optional[str] = None
    error: Optional[str] = None


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for coding agents.

    Implements LATS (Language Agent Tree Search) with:
    - Selection: UCB1 with exploration bonus
    - Expansion: LLM-generated action proposals
    - Simulation: LLM rollouts to terminal states
    - Backpropagation: Reward propagation with decay

    Integrates with Reflexion for learning from failures.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        max_depth: int = 10,
        num_simulations: int = 50,
        expansion_width: int = 3,
        exploration_constant: float = 1.414,  # sqrt(2)
        reward_decay: float = 0.95,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.expansion_width = expansion_width
        self.exploration_constant = exploration_constant
        self.reward_decay = reward_decay
        self.temperature = temperature

        self._node_counter = 0
        self._simulation_cache: Dict[str, SimulationResult] = {}

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"

    async def plan(
        self,
        task: str,
        initial_state: Dict[str, Any],
        reward_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        reflexion_memories: Optional[List[str]] = None,
    ) -> Tuple[List[MCTSNode], float]:
        """
        Plan solution using MCTS.

        Args:
            task: Task description
            initial_state: Initial problem state
            reward_fn: Optional custom reward function
            reflexion_memories: Past reflections to guide search

        Returns:
            Tuple of (best path, expected reward)
        """
        # Create root node
        root = MCTSNode(
            id=self._generate_node_id(),
            action="start",
            action_type=ActionType.THINK,
            state=initial_state,
        )

        # Add reflexion memories if provided
        if reflexion_memories:
            root.reflections.extend(reflexion_memories)

        # Run MCTS iterations
        for iteration in range(self.num_simulations):
            # Selection
            node = await self._select(root)

            # Check if we need to expand
            if not node.is_terminal and node.depth < self.max_depth:
                # Expansion
                if not node.children or (node.visits > 0 and len(node.children) < self.expansion_width):
                    await self._expand(node, task)

                # Select newly expanded child if available
                if node.children:
                    node = node.children[-1]

            # Simulation
            result = await self._simulate(node, task, reward_fn)

            # Backpropagation
            self._backpropagate(node, result.reward)

            # Store reflection if simulation failed
            if not result.success and result.reflection:
                node.add_reflection(result.reflection)

        # Extract best path
        best_path = self._extract_best_path(root)
        expected_reward = best_path[-1].average_reward if best_path else 0.0

        return best_path, expected_reward

    async def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Select node for expansion using UCB1.

        Traverses tree selecting children with highest UCB1 score
        until reaching a leaf node or unexpanded node.
        """
        node = root

        while node.children and not node.is_terminal:
            # Select child with highest UCB1 score
            best_child = max(node.children, key=lambda c: c.ucb1_score)

            # If best child is unexplored, select it
            if best_child.visits == 0:
                return best_child

            node = best_child

        return node

    async def _expand(self, node: MCTSNode, task: str) -> List[MCTSNode]:
        """
        Expand node by generating possible actions.

        Uses LLM to propose actions based on current state
        and any reflections from past failures.
        """
        # Build prompt with state and reflections
        reflection_context = ""
        if node.reflections:
            reflection_context = "\n\nLearnings from past attempts:\n" + "\n".join(
                f"- {r}" for r in node.reflections[-3:]  # Last 3 reflections
            )

        prompt = f"""You are planning the next action for a coding task.

Task: {task}

Current state:
{self._format_state(node.state)}

Path so far:
{self._format_path(node.get_path())}
{reflection_context}

Propose {self.expansion_width} distinct next actions to make progress.
For each action, specify:
1. Action type: one of [think, code, test, search, read, execute, verify, refine]
2. Action description: what specifically to do

Format each action as:
ACTION_TYPE: description

Actions:"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # Parse actions from response
        actions = self._parse_actions(response.choices[0].message.content)

        # Create child nodes
        new_children = []
        for action_type, action_desc in actions[:self.expansion_width]:
            child = MCTSNode(
                id=self._generate_node_id(),
                action=action_desc,
                action_type=action_type,
                state=node.state.copy(),  # State will be updated during simulation
            )
            node.add_child(child)
            new_children.append(child)

        return new_children

    async def _simulate(
        self,
        node: MCTSNode,
        task: str,
        reward_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> SimulationResult:
        """
        Simulate from node to terminal state.

        Performs LLM rollout, executing actions and evaluating
        the resulting state for reward computation.
        """
        # Check cache
        cache_key = f"{node.id}_{hash(str(node.state))}"
        if cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]

        trajectory = []
        current_state = node.state.copy()
        current_depth = node.depth

        # Rollout until terminal or max depth
        while current_depth < self.max_depth:
            # Generate next action via LLM
            prompt = f"""Continue solving this coding task.

Task: {task}

Current state:
{self._format_state(current_state)}

Trajectory so far:
{chr(10).join(trajectory) if trajectory else 'None'}

What is the next action? If the task is complete, respond with "DONE: <summary>".
Otherwise, respond with the action to take."""

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            action = response.choices[0].message.content.strip()
            trajectory.append(action)

            # Check for terminal
            if action.startswith("DONE:"):
                break

            # Update state (simplified - in practice would execute action)
            current_state["last_action"] = action
            current_state["action_count"] = current_state.get("action_count", 0) + 1
            current_depth += 1

        # Compute reward
        if reward_fn:
            reward = reward_fn(current_state)
        else:
            reward = await self._compute_reward(task, current_state, trajectory)

        # Generate reflection if reward is low
        reflection = None
        if reward < 0.5:
            reflection = await self._generate_reflection(task, trajectory, current_state)

        result = SimulationResult(
            success=reward >= 0.7,
            reward=reward,
            trajectory=trajectory,
            final_state=current_state,
            reflection=reflection,
        )

        # Cache result
        self._simulation_cache[cache_key] = result

        return result

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate reward through tree.

        Updates visit counts and total rewards for all nodes
        in the path from the simulated node to root.
        """
        current = node
        current_reward = reward

        while current is not None:
            current.visits += 1
            current.total_reward += current_reward

            # Decay reward as we go up the tree
            current_reward *= self.reward_decay
            current = current.parent

    async def _compute_reward(
        self,
        task: str,
        state: Dict[str, Any],
        trajectory: List[str],
    ) -> float:
        """Compute reward for a terminal state using LLM evaluation."""
        prompt = f"""Evaluate how well this solution attempt addresses the task.

Task: {task}

Actions taken:
{chr(10).join(f'{i+1}. {a}' for i, a in enumerate(trajectory))}

Final state:
{self._format_state(state)}

Rate the solution from 0.0 (complete failure) to 1.0 (perfect solution).
Consider:
- Task completion: Was the objective achieved?
- Code quality: Is the solution clean and maintainable?
- Efficiency: Is the approach reasonable?

Respond with just a number between 0.0 and 1.0."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent evaluation
        )

        try:
            reward = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
        except ValueError:
            return 0.5  # Default reward on parse failure

    async def _generate_reflection(
        self,
        task: str,
        trajectory: List[str],
        state: Dict[str, Any],
    ) -> str:
        """Generate reflection from failed attempt for learning."""
        prompt = f"""Analyze why this attempt did not fully solve the task.

Task: {task}

Actions taken:
{chr(10).join(f'{i+1}. {a}' for i, a in enumerate(trajectory))}

What went wrong? What should be done differently next time?
Provide a concise reflection (1-2 sentences) that captures the key insight."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def _extract_best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Extract the best path through the tree based on average reward."""
        path = [root]
        current = root

        while current.children:
            # Select child with highest average reward (exploitation only)
            best_child = max(current.children, key=lambda c: c.average_reward)
            path.append(best_child)
            current = best_child

        return path

    def _parse_actions(self, text: str) -> List[Tuple[ActionType, str]]:
        """Parse actions from LLM response."""
        actions = []

        action_map = {
            "think": ActionType.THINK,
            "code": ActionType.CODE,
            "test": ActionType.TEST,
            "search": ActionType.SEARCH,
            "read": ActionType.READ,
            "execute": ActionType.EXECUTE,
            "verify": ActionType.VERIFY,
            "refine": ActionType.REFINE,
        }

        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                action_type_str = parts[0].strip().lower()
                action_desc = parts[1].strip() if len(parts) > 1 else ""

                if action_type_str in action_map:
                    actions.append((action_map[action_type_str], action_desc))

        # Default to THINK actions if parsing fails
        if not actions:
            actions = [(ActionType.THINK, "Analyze the problem")]

        return actions

    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state dictionary for prompt."""
        lines = []
        for key, value in state.items():
            if isinstance(value, (list, dict)):
                lines.append(f"  {key}: {len(value)} items")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "  (empty)"

    def _format_path(self, path: List[MCTSNode]) -> str:
        """Format action path for prompt."""
        if len(path) <= 1:
            return "  (just started)"

        lines = []
        for i, node in enumerate(path[1:], 1):  # Skip root
            lines.append(f"  {i}. [{node.action_type.value}] {node.action}")
        return "\n".join(lines)

    def get_tree_stats(self, root: MCTSNode) -> Dict[str, Any]:
        """Get statistics about the search tree."""
        total_nodes = 0
        total_visits = 0
        max_depth = 0
        terminal_nodes = 0

        def traverse(node: MCTSNode):
            nonlocal total_nodes, total_visits, max_depth, terminal_nodes
            total_nodes += 1
            total_visits += node.visits
            max_depth = max(max_depth, node.depth)
            if node.is_terminal:
                terminal_nodes += 1
            for child in node.children:
                traverse(child)

        traverse(root)

        return {
            "total_nodes": total_nodes,
            "total_visits": total_visits,
            "max_depth": max_depth,
            "terminal_nodes": terminal_nodes,
            "average_visits": total_visits / total_nodes if total_nodes > 0 else 0,
        }
