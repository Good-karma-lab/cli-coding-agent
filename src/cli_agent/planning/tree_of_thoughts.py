"""
Tree of Thoughts (ToT) - Multi-path Deliberative Reasoning

Based on research for complex problem solving:
- Generates multiple reasoning paths simultaneously
- Evaluates thought quality at each step
- Prunes unpromising branches
- Supports BFS, DFS, and beam search strategies
- Integrates with MCTS for combined exploration
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
import litellm
from pydantic import BaseModel


class SearchStrategy(str, Enum):
    """Search strategies for Tree of Thoughts."""
    BFS = "bfs"      # Breadth-first: explore all thoughts at each depth
    DFS = "dfs"      # Depth-first: follow promising paths deeply
    BEAM = "beam"    # Beam search: keep top-k thoughts at each level


class ThoughtState(str, Enum):
    """State of a thought node."""
    PENDING = "pending"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    SELECTED = "selected"
    TERMINAL = "terminal"


@dataclass
class ThoughtNode:
    """
    Node in the Tree of Thoughts.

    Represents a single thought/reasoning step with
    evaluation scores for quality assessment.
    """
    id: str
    thought: str
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)

    # Evaluation
    state: ThoughtState = ThoughtState.PENDING
    value_score: float = 0.0         # How promising is this thought
    coherence_score: float = 0.0     # How coherent with previous thoughts
    novelty_score: float = 0.0       # How novel/non-redundant
    feasibility_score: float = 0.0   # How executable/practical

    # Metadata
    depth: int = 0
    is_solution: bool = False
    solution_quality: Optional[float] = None

    @property
    def combined_score(self) -> float:
        """Combined evaluation score."""
        return (
            0.4 * self.value_score +
            0.2 * self.coherence_score +
            0.2 * self.novelty_score +
            0.2 * self.feasibility_score
        )

    def add_child(self, child: "ThoughtNode") -> None:
        """Add a child thought."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def get_path(self) -> List["ThoughtNode"]:
        """Get reasoning path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_thought_chain(self) -> str:
        """Get chain of thoughts as text."""
        path = self.get_path()
        return "\n".join(f"Step {i+1}: {n.thought}" for i, n in enumerate(path))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "thought": self.thought,
            "state": self.state.value,
            "value_score": self.value_score,
            "coherence_score": self.coherence_score,
            "combined_score": self.combined_score,
            "depth": self.depth,
            "is_solution": self.is_solution,
            "num_children": len(self.children),
        }


class ThoughtEvaluation(BaseModel):
    """Evaluation result for a thought."""
    value_score: float
    coherence_score: float
    novelty_score: float
    feasibility_score: float
    reasoning: str
    is_promising: bool


class TreeOfThoughts:
    """
    Tree of Thoughts planner for deliberative reasoning.

    Generates and evaluates multiple reasoning paths,
    pruning unpromising branches and selecting the best
    path to a solution.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        strategy: SearchStrategy = SearchStrategy.BEAM,
        max_depth: int = 5,
        branching_factor: int = 3,
        beam_width: int = 3,
        pruning_threshold: float = 0.3,
        temperature: float = 0.8,
    ):
        self.model = model
        self.strategy = strategy
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.beam_width = beam_width
        self.pruning_threshold = pruning_threshold
        self.temperature = temperature

        self._node_counter = 0
        self._all_thoughts: Set[str] = set()  # For novelty checking

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"thought_{self._node_counter}"

    async def solve(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        validator: Optional[Callable[[str], bool]] = None,
    ) -> List[ThoughtNode]:
        """
        Solve a problem using Tree of Thoughts.

        Args:
            problem: Problem description
            context: Additional context
            validator: Optional function to validate solutions

        Returns:
            List of thought nodes representing best reasoning path
        """
        self._all_thoughts.clear()

        # Create root node
        root = ThoughtNode(
            id=self._generate_node_id(),
            thought=f"Problem: {problem}",
        )
        root.state = ThoughtState.EVALUATED

        # Run search based on strategy
        if self.strategy == SearchStrategy.BFS:
            solutions = await self._bfs_search(root, problem, context, validator)
        elif self.strategy == SearchStrategy.DFS:
            solutions = await self._dfs_search(root, problem, context, validator)
        else:  # BEAM
            solutions = await self._beam_search(root, problem, context, validator)

        # Return best solution path
        if solutions:
            best = max(solutions, key=lambda n: n.solution_quality or n.combined_score)
            return best.get_path()

        # Return deepest explored path if no solution found
        return self._get_deepest_path(root)

    async def _beam_search(
        self,
        root: ThoughtNode,
        problem: str,
        context: Optional[Dict[str, Any]],
        validator: Optional[Callable[[str], bool]],
    ) -> List[ThoughtNode]:
        """
        Beam search: keep top-k thoughts at each level.

        Most effective for problems with clear evaluation criteria.
        """
        solutions = []
        current_level = [root]

        for depth in range(self.max_depth):
            # Generate thoughts for all nodes at current level
            all_children = []

            generation_tasks = [
                self._generate_thoughts(node, problem, context)
                for node in current_level
                if node.state not in (ThoughtState.PRUNED, ThoughtState.TERMINAL)
            ]

            children_lists = await asyncio.gather(*generation_tasks)

            for children in children_lists:
                all_children.extend(children)

            if not all_children:
                break

            # Evaluate all children
            evaluation_tasks = [
                self._evaluate_thought(child, problem, context)
                for child in all_children
            ]
            await asyncio.gather(*evaluation_tasks)

            # Check for solutions
            for child in all_children:
                if await self._is_solution(child, problem, validator):
                    child.is_solution = True
                    child.state = ThoughtState.TERMINAL
                    child.solution_quality = await self._evaluate_solution(
                        child, problem, context
                    )
                    solutions.append(child)

            # Prune low-scoring thoughts
            viable_children = [
                c for c in all_children
                if c.combined_score >= self.pruning_threshold
                and c.state != ThoughtState.TERMINAL
            ]

            # Select top-k for next level
            viable_children.sort(key=lambda c: c.combined_score, reverse=True)
            current_level = viable_children[:self.beam_width]

            for node in current_level:
                node.state = ThoughtState.SELECTED

            if not current_level:
                break

        return solutions

    async def _bfs_search(
        self,
        root: ThoughtNode,
        problem: str,
        context: Optional[Dict[str, Any]],
        validator: Optional[Callable[[str], bool]],
    ) -> List[ThoughtNode]:
        """
        Breadth-first search: explore all thoughts at each depth.

        Thorough but potentially expensive.
        """
        solutions = []
        queue = [root]

        while queue:
            current = queue.pop(0)

            if current.depth >= self.max_depth:
                continue

            if current.state == ThoughtState.PRUNED:
                continue

            # Generate and evaluate children
            children = await self._generate_thoughts(current, problem, context)

            for child in children:
                await self._evaluate_thought(child, problem, context)

                if await self._is_solution(child, problem, validator):
                    child.is_solution = True
                    child.state = ThoughtState.TERMINAL
                    child.solution_quality = await self._evaluate_solution(
                        child, problem, context
                    )
                    solutions.append(child)
                elif child.combined_score >= self.pruning_threshold:
                    queue.append(child)
                else:
                    child.state = ThoughtState.PRUNED

        return solutions

    async def _dfs_search(
        self,
        root: ThoughtNode,
        problem: str,
        context: Optional[Dict[str, Any]],
        validator: Optional[Callable[[str], bool]],
    ) -> List[ThoughtNode]:
        """
        Depth-first search: follow promising paths deeply.

        Fast for problems with strong early signals.
        """
        solutions = []
        stack = [root]

        while stack:
            current = stack.pop()

            if current.depth >= self.max_depth:
                continue

            if current.state == ThoughtState.PRUNED:
                continue

            # Generate and evaluate children
            children = await self._generate_thoughts(current, problem, context)

            for child in children:
                await self._evaluate_thought(child, problem, context)

                if await self._is_solution(child, problem, validator):
                    child.is_solution = True
                    child.state = ThoughtState.TERMINAL
                    child.solution_quality = await self._evaluate_solution(
                        child, problem, context
                    )
                    solutions.append(child)
                elif child.combined_score >= self.pruning_threshold:
                    stack.append(child)
                else:
                    child.state = ThoughtState.PRUNED

            # Sort stack to explore most promising first
            stack.sort(key=lambda n: n.combined_score)

        return solutions

    async def _generate_thoughts(
        self,
        node: ThoughtNode,
        problem: str,
        context: Optional[Dict[str, Any]],
    ) -> List[ThoughtNode]:
        """Generate next thoughts from current node."""
        thought_chain = node.get_thought_chain()
        context_str = self._format_context(context) if context else ""

        prompt = f"""You are solving a problem step by step.

Problem: {problem}
{context_str}

Current reasoning chain:
{thought_chain}

Generate {self.branching_factor} distinct next thoughts to continue solving this problem.
Each thought should be a clear, concrete reasoning step that builds on the previous chain.

Thoughts should be diverse - explore different approaches or aspects of the problem.
If you believe the problem is solved, state "SOLUTION:" followed by the answer.

Format each thought on a separate line, numbered 1-{self.branching_factor}:"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # Parse thoughts
        thoughts = self._parse_thoughts(response.choices[0].message.content)

        # Create child nodes
        children = []
        for thought in thoughts[:self.branching_factor]:
            # Skip duplicate thoughts
            thought_key = thought.lower().strip()
            if thought_key in self._all_thoughts:
                continue
            self._all_thoughts.add(thought_key)

            child = ThoughtNode(
                id=self._generate_node_id(),
                thought=thought,
            )
            node.add_child(child)
            children.append(child)

        return children

    async def _evaluate_thought(
        self,
        node: ThoughtNode,
        problem: str,
        context: Optional[Dict[str, Any]],
    ) -> ThoughtEvaluation:
        """Evaluate a thought node."""
        thought_chain = node.get_thought_chain()
        context_str = self._format_context(context) if context else ""

        prompt = f"""Evaluate this reasoning step for solving a problem.

Problem: {problem}
{context_str}

Reasoning chain:
{thought_chain}

Evaluate the latest thought on these criteria (0.0 to 1.0):
1. VALUE: Does this thought make progress toward solving the problem?
2. COHERENCE: Is it logically consistent with previous thoughts?
3. NOVELTY: Does it add new information (not redundant)?
4. FEASIBILITY: Is this thought actionable and practical?

Also determine if this thought chain is promising enough to continue exploring.

Format your response as:
VALUE: <score>
COHERENCE: <score>
NOVELTY: <score>
FEASIBILITY: <score>
PROMISING: <yes/no>
REASONING: <brief explanation>"""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Low temperature for consistent evaluation
        )

        # Parse evaluation
        eval_result = self._parse_evaluation(response.choices[0].message.content)

        # Update node
        node.value_score = eval_result.value_score
        node.coherence_score = eval_result.coherence_score
        node.novelty_score = eval_result.novelty_score
        node.feasibility_score = eval_result.feasibility_score
        node.state = ThoughtState.EVALUATED

        return eval_result

    async def _is_solution(
        self,
        node: ThoughtNode,
        problem: str,
        validator: Optional[Callable[[str], bool]],
    ) -> bool:
        """Check if a thought represents a complete solution."""
        # Check for explicit solution marker
        if "SOLUTION:" in node.thought.upper():
            if validator:
                return validator(node.thought)
            return True

        # Ask LLM to check
        prompt = f"""Does this reasoning chain completely solve the problem?

Problem: {problem}

Reasoning chain:
{node.get_thought_chain()}

Answer with just YES or NO."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip().upper()
        is_complete = "YES" in answer

        if is_complete and validator:
            return validator(node.thought)

        return is_complete

    async def _evaluate_solution(
        self,
        node: ThoughtNode,
        problem: str,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Evaluate the quality of a solution."""
        prompt = f"""Rate the quality of this solution.

Problem: {problem}

Solution reasoning:
{node.get_thought_chain()}

Rate from 0.0 (incorrect) to 1.0 (perfect solution).
Consider correctness, completeness, and elegance.

Respond with just a number."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    def _get_deepest_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Get the deepest path in the tree."""
        deepest = root
        max_depth = 0

        def traverse(node: ThoughtNode):
            nonlocal deepest, max_depth
            if node.depth > max_depth:
                max_depth = node.depth
                deepest = node
            for child in node.children:
                traverse(child)

        traverse(root)
        return deepest.get_path()

    def _parse_thoughts(self, text: str) -> List[str]:
        """Parse thoughts from LLM response."""
        thoughts = []
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Remove numbering
            if line and line[0].isdigit():
                # Remove "1.", "1)", "1:" etc.
                for sep in [".", ")", ":"]:
                    if sep in line:
                        idx = line.index(sep)
                        if idx < 3:  # Likely a number prefix
                            line = line[idx + 1:].strip()
                            break

            if line and len(line) > 5:  # Non-trivial thought
                thoughts.append(line)

        return thoughts

    def _parse_evaluation(self, text: str) -> ThoughtEvaluation:
        """Parse evaluation from LLM response."""
        scores = {
            "value": 0.5,
            "coherence": 0.5,
            "novelty": 0.5,
            "feasibility": 0.5,
        }
        is_promising = True
        reasoning = ""

        lines = text.strip().split("\n")
        for line in lines:
            line_lower = line.lower()
            if "value:" in line_lower:
                scores["value"] = self._extract_score(line)
            elif "coherence:" in line_lower:
                scores["coherence"] = self._extract_score(line)
            elif "novelty:" in line_lower:
                scores["novelty"] = self._extract_score(line)
            elif "feasibility:" in line_lower:
                scores["feasibility"] = self._extract_score(line)
            elif "promising:" in line_lower:
                is_promising = "yes" in line_lower
            elif "reasoning:" in line_lower:
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""

        return ThoughtEvaluation(
            value_score=scores["value"],
            coherence_score=scores["coherence"],
            novelty_score=scores["novelty"],
            feasibility_score=scores["feasibility"],
            reasoning=reasoning,
            is_promising=is_promising,
        )

    def _extract_score(self, line: str) -> float:
        """Extract numeric score from a line."""
        import re
        match = re.search(r"(\d+\.?\d*)", line)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompts."""
        lines = ["Context:"]
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                lines.append(f"  {key}: {len(value)} items")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def get_tree_stats(self, root: ThoughtNode) -> Dict[str, Any]:
        """Get statistics about the thought tree."""
        total_nodes = 0
        evaluated_nodes = 0
        pruned_nodes = 0
        solutions = 0
        max_depth = 0
        total_score = 0.0

        def traverse(node: ThoughtNode):
            nonlocal total_nodes, evaluated_nodes, pruned_nodes
            nonlocal solutions, max_depth, total_score

            total_nodes += 1
            max_depth = max(max_depth, node.depth)
            total_score += node.combined_score

            if node.state == ThoughtState.EVALUATED:
                evaluated_nodes += 1
            elif node.state == ThoughtState.PRUNED:
                pruned_nodes += 1

            if node.is_solution:
                solutions += 1

            for child in node.children:
                traverse(child)

        traverse(root)

        return {
            "total_nodes": total_nodes,
            "evaluated_nodes": evaluated_nodes,
            "pruned_nodes": pruned_nodes,
            "solutions_found": solutions,
            "max_depth": max_depth,
            "average_score": total_score / total_nodes if total_nodes > 0 else 0,
            "pruning_rate": pruned_nodes / total_nodes if total_nodes > 0 else 0,
        }
