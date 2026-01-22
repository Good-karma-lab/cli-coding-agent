"""
Multi-Island Evolutionary Strategy with MAP-Elites

Based on research for parallel agent exploration:
- Multiple islands explore solution space independently
- Periodic migration of elite solutions between islands
- MAP-Elites archive for quality-diversity optimization
- Adaptive mutation strategies per island
- Supports parallel execution across git worktrees
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
from enum import Enum
import litellm
from pydantic import BaseModel
import hashlib


class MutationType(str, Enum):
    """Types of mutations for solutions."""
    REFINE = "refine"            # Small refinements
    RESTRUCTURE = "restructure"  # Larger structural changes
    CROSSOVER = "crossover"      # Combine with another solution
    SIMPLIFY = "simplify"        # Remove complexity
    EXTEND = "extend"            # Add new components
    SPECIALIZE = "specialize"    # Focus on specific aspect


class IslandStrategy(str, Enum):
    """Strategy focus for each island."""
    EXPLORATION = "exploration"   # Try diverse approaches
    EXPLOITATION = "exploitation" # Refine best solutions
    QUALITY = "quality"           # Focus on code quality
    EFFICIENCY = "efficiency"     # Focus on performance
    SAFETY = "safety"             # Focus on correctness
    CREATIVITY = "creativity"     # Try unconventional approaches


@dataclass
class Solution:
    """
    A candidate solution in the evolutionary process.

    Represents a complete approach to solving a task,
    with associated fitness scores and lineage tracking.
    """
    id: str
    content: str  # The actual solution (code, plan, etc.)
    fitness: float = 0.0

    # Multi-dimensional quality scores for MAP-Elites
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # Behavior characterization for diversity
    behavior_descriptor: Tuple[float, ...] = field(default_factory=tuple)

    # Lineage tracking
    parent_ids: List[str] = field(default_factory=list)
    mutation_type: Optional[MutationType] = None
    generation: int = 0

    # Metadata
    island_id: Optional[str] = None
    created_at: Optional[float] = None
    evaluation_count: int = 0

    def __hash__(self):
        return hash(self.id)

    def get_descriptor_key(self, grid_resolution: int = 10) -> Tuple[int, ...]:
        """Get discretized behavior descriptor for MAP-Elites grid."""
        if not self.behavior_descriptor:
            return (0,)
        return tuple(
            min(grid_resolution - 1, int(v * grid_resolution))
            for v in self.behavior_descriptor
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "fitness": self.fitness,
            "quality_scores": self.quality_scores,
            "behavior_descriptor": self.behavior_descriptor,
            "generation": self.generation,
            "island_id": self.island_id,
            "mutation_type": self.mutation_type.value if self.mutation_type else None,
        }


class MAPElitesArchive:
    """
    MAP-Elites archive for quality-diversity optimization.

    Maintains a grid of elite solutions, where each cell
    represents a unique behavior region. Only the best
    solution in each region is kept.
    """

    def __init__(
        self,
        dimensions: int = 2,
        grid_resolution: int = 10,
    ):
        self.dimensions = dimensions
        self.grid_resolution = grid_resolution
        self._archive: Dict[Tuple[int, ...], Solution] = {}
        self._all_solutions: List[Solution] = []

    def add(self, solution: Solution) -> bool:
        """
        Add solution to archive if it improves its cell.

        Returns True if solution was added.
        """
        key = solution.get_descriptor_key(self.grid_resolution)
        self._all_solutions.append(solution)

        if key not in self._archive:
            self._archive[key] = solution
            return True

        if solution.fitness > self._archive[key].fitness:
            self._archive[key] = solution
            return True

        return False

    def get_elites(self) -> List[Solution]:
        """Get all elite solutions."""
        return list(self._archive.values())

    def get_best(self, n: int = 1) -> List[Solution]:
        """Get top n solutions by fitness."""
        elites = self.get_elites()
        elites.sort(key=lambda s: s.fitness, reverse=True)
        return elites[:n]

    def get_diverse(self, n: int = 5) -> List[Solution]:
        """Get n diverse solutions from different cells."""
        elites = self.get_elites()
        if len(elites) <= n:
            return elites

        # Sample from different regions
        selected = []
        available = list(elites)
        random.shuffle(available)

        for solution in available:
            if len(selected) >= n:
                break
            # Check if this region is covered
            key = solution.get_descriptor_key(self.grid_resolution)
            if not any(
                s.get_descriptor_key(self.grid_resolution) == key
                for s in selected
            ):
                selected.append(solution)

        # Fill remaining with best
        if len(selected) < n:
            remaining = [s for s in elites if s not in selected]
            remaining.sort(key=lambda s: s.fitness, reverse=True)
            selected.extend(remaining[:n - len(selected)])

        return selected

    def coverage(self) -> float:
        """Get archive coverage (filled cells / total cells)."""
        total_cells = self.grid_resolution ** self.dimensions
        return len(self._archive) / total_cells

    def stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        elites = self.get_elites()
        fitnesses = [s.fitness for s in elites]

        return {
            "num_elites": len(elites),
            "coverage": self.coverage(),
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            "total_evaluated": len(self._all_solutions),
        }


@dataclass
class Island:
    """
    An island in the multi-island model.

    Each island maintains its own population and
    evolves independently with a specific strategy.
    """
    id: str
    strategy: IslandStrategy
    population: List[Solution] = field(default_factory=list)
    archive: MAPElitesArchive = field(default_factory=MAPElitesArchive)

    # Evolution parameters
    population_size: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_ratio: float = 0.2

    # Statistics
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    improvements: int = 0

    def add_solution(self, solution: Solution) -> None:
        """Add solution to island."""
        solution.island_id = self.id
        self.population.append(solution)
        self.archive.add(solution)

    def get_elites(self, n: Optional[int] = None) -> List[Solution]:
        """Get elite solutions from population."""
        n = n or int(len(self.population) * self.elite_ratio)
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)
        return sorted_pop[:max(1, n)]

    def select_parents(self, n: int = 2) -> List[Solution]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        for _ in range(n):
            tournament = random.sample(
                self.population,
                min(3, len(self.population))
            )
            winner = max(tournament, key=lambda s: s.fitness)
            parents.append(winner)
        return parents

    def update_stats(self) -> None:
        """Update island statistics."""
        if self.population:
            fitnesses = [s.fitness for s in self.population]
            self.best_fitness = max(fitnesses)
            self.avg_fitness = sum(fitnesses) / len(fitnesses)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "population_size": len(self.population),
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "archive_coverage": self.archive.coverage(),
        }


class MultiIslandEvolution:
    """
    Multi-Island evolutionary model for solution generation.

    Features:
    - Multiple islands with different strategies
    - Periodic migration of elite solutions
    - MAP-Elites quality-diversity archive
    - Adaptive mutation based on island strategy
    - Support for parallel evaluation
    """

    def __init__(
        self,
        model: str = "gpt-4",
        num_islands: int = 4,
        population_per_island: int = 10,
        migration_interval: int = 5,
        migration_size: int = 2,
        max_generations: int = 20,
        behavior_dimensions: int = 2,
    ):
        self.model = model
        self.num_islands = num_islands
        self.population_per_island = population_per_island
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.max_generations = max_generations
        self.behavior_dimensions = behavior_dimensions

        self._solution_counter = 0
        self.islands: List[Island] = []
        self.global_archive = MAPElitesArchive(
            dimensions=behavior_dimensions,
            grid_resolution=20,
        )

    def _generate_solution_id(self) -> str:
        """Generate unique solution ID."""
        self._solution_counter += 1
        return f"sol_{self._solution_counter}"

    async def evolve(
        self,
        task: str,
        evaluator: Callable[[Solution, str], Tuple[float, Dict[str, float]]],
        initial_solutions: Optional[List[str]] = None,
    ) -> List[Solution]:
        """
        Run evolutionary optimization.

        Args:
            task: Task description
            evaluator: Function to evaluate solutions
            initial_solutions: Optional seed solutions

        Returns:
            List of elite solutions
        """
        # Initialize islands with different strategies
        strategies = list(IslandStrategy)
        self.islands = [
            Island(
                id=f"island_{i}",
                strategy=strategies[i % len(strategies)],
                population_size=self.population_per_island,
                archive=MAPElitesArchive(dimensions=self.behavior_dimensions),
            )
            for i in range(self.num_islands)
        ]

        # Initialize populations
        await self._initialize_populations(task, initial_solutions)

        # Evaluate initial populations
        await self._evaluate_all(task, evaluator)

        # Evolution loop
        for gen in range(self.max_generations):
            # Evolve each island
            evolution_tasks = [
                self._evolve_island(island, task, evaluator)
                for island in self.islands
            ]
            await asyncio.gather(*evolution_tasks)

            # Periodic migration
            if (gen + 1) % self.migration_interval == 0:
                await self._migrate()

            # Update generation count
            for island in self.islands:
                island.generation = gen + 1
                island.update_stats()

        # Return best solutions from global archive
        return self.global_archive.get_best(10)

    async def _initialize_populations(
        self,
        task: str,
        initial_solutions: Optional[List[str]],
    ) -> None:
        """Initialize island populations."""
        # Generate initial solutions for each island
        for island in self.islands:
            if initial_solutions:
                # Distribute initial solutions
                for i, content in enumerate(initial_solutions):
                    if i % self.num_islands == self.islands.index(island):
                        solution = Solution(
                            id=self._generate_solution_id(),
                            content=content,
                            generation=0,
                        )
                        island.add_solution(solution)

            # Fill remaining slots with generated solutions
            while len(island.population) < island.population_size:
                content = await self._generate_solution(
                    task, island.strategy, existing=island.population
                )
                solution = Solution(
                    id=self._generate_solution_id(),
                    content=content,
                    generation=0,
                )
                island.add_solution(solution)

    async def _generate_solution(
        self,
        task: str,
        strategy: IslandStrategy,
        existing: List[Solution],
    ) -> str:
        """Generate a new solution based on strategy."""
        strategy_prompts = {
            IslandStrategy.EXPLORATION: "Try an unconventional or creative approach",
            IslandStrategy.EXPLOITATION: "Build on established patterns and best practices",
            IslandStrategy.QUALITY: "Focus on clean, maintainable, well-documented code",
            IslandStrategy.EFFICIENCY: "Optimize for performance and resource usage",
            IslandStrategy.SAFETY: "Prioritize correctness, error handling, and security",
            IslandStrategy.CREATIVITY: "Explore novel algorithms or architectures",
        }

        existing_summary = ""
        if existing:
            existing_summary = f"\n\nExisting approaches to avoid duplicating:\n" + "\n".join(
                f"- {s.content[:100]}..." for s in existing[-3:]
            )

        prompt = f"""Generate a solution for this task.

Task: {task}

Strategy: {strategy_prompts.get(strategy, 'Generate a good solution')}
{existing_summary}

Provide a complete, working solution. Be specific and detailed."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )

        return response.choices[0].message.content.strip()

    async def _evaluate_all(
        self,
        task: str,
        evaluator: Callable[[Solution, str], Tuple[float, Dict[str, float]]],
    ) -> None:
        """Evaluate all solutions across islands."""
        for island in self.islands:
            for solution in island.population:
                if solution.evaluation_count == 0:
                    fitness, quality_scores = await asyncio.to_thread(
                        evaluator, solution, task
                    )
                    solution.fitness = fitness
                    solution.quality_scores = quality_scores
                    solution.evaluation_count += 1

                    # Compute behavior descriptor
                    solution.behavior_descriptor = self._compute_behavior(
                        solution, quality_scores
                    )

                    # Add to global archive
                    self.global_archive.add(solution)

    async def _evolve_island(
        self,
        island: Island,
        task: str,
        evaluator: Callable[[Solution, str], Tuple[float, Dict[str, float]]],
    ) -> None:
        """Evolve a single island for one generation."""
        new_population = []

        # Keep elites
        elites = island.get_elites()
        new_population.extend(elites)

        # Generate offspring
        while len(new_population) < island.population_size:
            parents = island.select_parents(2)

            # Decide mutation type based on strategy
            mutation_type = self._select_mutation_type(island.strategy)

            if mutation_type == MutationType.CROSSOVER and len(parents) >= 2:
                offspring_content = await self._crossover(
                    parents[0], parents[1], task
                )
                parent_ids = [p.id for p in parents]
            else:
                offspring_content = await self._mutate(
                    parents[0], mutation_type, task, island.strategy
                )
                parent_ids = [parents[0].id]

            offspring = Solution(
                id=self._generate_solution_id(),
                content=offspring_content,
                parent_ids=parent_ids,
                mutation_type=mutation_type,
                generation=island.generation + 1,
            )

            # Evaluate offspring
            fitness, quality_scores = await asyncio.to_thread(
                evaluator, offspring, task
            )
            offspring.fitness = fitness
            offspring.quality_scores = quality_scores
            offspring.evaluation_count = 1
            offspring.behavior_descriptor = self._compute_behavior(
                offspring, quality_scores
            )

            island.add_solution(offspring)
            new_population.append(offspring)

            # Track improvements
            if offspring.fitness > island.best_fitness:
                island.improvements += 1

            # Add to global archive
            self.global_archive.add(offspring)

        # Update population
        island.population = new_population

    def _select_mutation_type(self, strategy: IslandStrategy) -> MutationType:
        """Select mutation type based on island strategy."""
        type_weights = {
            IslandStrategy.EXPLORATION: {
                MutationType.RESTRUCTURE: 0.3,
                MutationType.EXTEND: 0.3,
                MutationType.CROSSOVER: 0.2,
                MutationType.REFINE: 0.2,
            },
            IslandStrategy.EXPLOITATION: {
                MutationType.REFINE: 0.5,
                MutationType.CROSSOVER: 0.3,
                MutationType.SIMPLIFY: 0.2,
            },
            IslandStrategy.QUALITY: {
                MutationType.REFINE: 0.4,
                MutationType.SIMPLIFY: 0.3,
                MutationType.RESTRUCTURE: 0.3,
            },
            IslandStrategy.EFFICIENCY: {
                MutationType.SIMPLIFY: 0.4,
                MutationType.REFINE: 0.3,
                MutationType.RESTRUCTURE: 0.3,
            },
            IslandStrategy.SAFETY: {
                MutationType.REFINE: 0.4,
                MutationType.SPECIALIZE: 0.3,
                MutationType.EXTEND: 0.3,
            },
            IslandStrategy.CREATIVITY: {
                MutationType.RESTRUCTURE: 0.3,
                MutationType.EXTEND: 0.3,
                MutationType.CROSSOVER: 0.4,
            },
        }

        weights = type_weights.get(strategy, {MutationType.REFINE: 1.0})
        types = list(weights.keys())
        probs = list(weights.values())

        return random.choices(types, weights=probs)[0]

    async def _mutate(
        self,
        solution: Solution,
        mutation_type: MutationType,
        task: str,
        strategy: IslandStrategy,
    ) -> str:
        """Apply mutation to a solution."""
        mutation_prompts = {
            MutationType.REFINE: "Make small refinements to improve this solution",
            MutationType.RESTRUCTURE: "Restructure this solution with a different architecture",
            MutationType.SIMPLIFY: "Simplify this solution by removing unnecessary complexity",
            MutationType.EXTEND: "Extend this solution with additional functionality",
            MutationType.SPECIALIZE: "Specialize this solution for better handling of edge cases",
        }

        prompt = f"""Modify this solution for a task.

Task: {task}

Current solution:
{solution.content}

Modification goal: {mutation_prompts.get(mutation_type, 'Improve the solution')}

Provide the modified solution. Keep what works well, change what can be improved."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    async def _crossover(
        self,
        parent1: Solution,
        parent2: Solution,
        task: str,
    ) -> str:
        """Combine two solutions."""
        prompt = f"""Combine the best aspects of two solutions.

Task: {task}

Solution 1:
{parent1.content}

Solution 2:
{parent2.content}

Create a new solution that combines the strengths of both.
Take the best ideas from each and merge them effectively."""

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    def _compute_behavior(
        self,
        solution: Solution,
        quality_scores: Dict[str, float],
    ) -> Tuple[float, ...]:
        """Compute behavior descriptor for diversity."""
        # Use quality scores as behavior dimensions
        dimensions = ["complexity", "coverage", "efficiency", "readability"]

        descriptor = []
        for dim in dimensions[:self.behavior_dimensions]:
            value = quality_scores.get(dim, 0.5)
            descriptor.append(max(0.0, min(1.0, value)))

        # Pad if needed
        while len(descriptor) < self.behavior_dimensions:
            # Use content hash for additional dimensions
            hash_val = int(hashlib.md5(solution.content.encode()).hexdigest()[:8], 16)
            descriptor.append((hash_val % 100) / 100.0)

        return tuple(descriptor[:self.behavior_dimensions])

    async def _migrate(self) -> None:
        """Migrate elite solutions between islands."""
        # Collect migrants from each island
        migrants: Dict[str, List[Solution]] = {}
        for island in self.islands:
            elites = island.archive.get_diverse(self.migration_size)
            migrants[island.id] = elites

        # Distribute migrants to other islands
        for i, island in enumerate(self.islands):
            # Receive from adjacent islands (ring topology)
            source_idx = (i - 1) % len(self.islands)
            source_island = self.islands[source_idx]

            for migrant in migrants.get(source_island.id, []):
                # Clone migrant for destination island
                clone = Solution(
                    id=self._generate_solution_id(),
                    content=migrant.content,
                    fitness=migrant.fitness,
                    quality_scores=migrant.quality_scores.copy(),
                    behavior_descriptor=migrant.behavior_descriptor,
                    parent_ids=[migrant.id],
                    generation=migrant.generation,
                )
                island.add_solution(clone)

    def get_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            "islands": [island.to_dict() for island in self.islands],
            "global_archive": self.global_archive.stats(),
            "total_solutions": self._solution_counter,
        }

    def get_best_solutions(self, n: int = 5) -> List[Solution]:
        """Get best solutions across all islands."""
        return self.global_archive.get_best(n)

    def get_diverse_solutions(self, n: int = 5) -> List[Solution]:
        """Get diverse solutions from global archive."""
        return self.global_archive.get_diverse(n)
