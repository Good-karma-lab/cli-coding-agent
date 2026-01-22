"""
Memory Manager - Unified interface for all memory systems.

Coordinates SimpleMem, A-MEM, AgeMem, and Episodic Memory.
Implements hierarchical memory separation between orchestrator and subagents.
"""

from datetime import datetime
from typing import Any, Optional

from cli_agent.core.config import MemoryConfig
from cli_agent.core.state import MemoryUnit, MemoryUnitType
from cli_agent.memory.simple_mem import SimpleMem
from cli_agent.memory.a_mem import AMem
from cli_agent.memory.age_mem import AgeMem
from cli_agent.memory.episodic import EpisodicMemory, Episode


class MemoryManager:
    """
    Unified Memory Manager coordinating all memory systems.

    Key principles from research:
    - Hierarchical separation: orchestrator vs subagent contexts
    - Large data as Artifacts with handles, not in context
    - Subagents have scoped views (suppress ancestral history)
    - 67% token reduction via parallel subagents with context isolation
    """

    def __init__(
        self,
        config: MemoryConfig,
        llm_client: Any,
        embedding_client: Any,
    ):
        self.config = config
        self.llm = llm_client
        self.embedder = embedding_client

        # Initialize memory systems
        self.simple_mem = SimpleMem(
            llm_client=llm_client,
            embedding_client=embedding_client,
            entropy_threshold=config.simple_mem.entropy_threshold,
            similarity_threshold=config.simple_mem.similarity_threshold,
            min_retrieval=config.simple_mem.min_retrieval_scope,
            max_retrieval=config.simple_mem.max_retrieval_scope,
        ) if config.simple_mem.enabled else None

        self.a_mem = AMem(
            embedding_client=embedding_client,
            link_threshold=config.a_mem.link_threshold,
            max_memories=config.a_mem.max_memory_units,
            enable_dynamic_linking=config.a_mem.enable_dynamic_linking,
        ) if config.a_mem.enabled else None

        self.age_mem = AgeMem(
            llm_client=llm_client,
            embedding_client=embedding_client,
            stm_capacity=config.age_mem.stm_capacity,
            ltm_capacity=config.age_mem.ltm_capacity,
            use_rl_policy=config.age_mem.use_rl_policy,
        ) if config.age_mem.enabled else None

        self.episodic = EpisodicMemory()

        # Artifact storage (handles for large data)
        self.artifacts: dict[str, str] = {}  # artifact_id -> file_path

        # Session insights (extracted during consolidation)
        self.session_insights: list[str] = []

    # =========================================================================
    # High-Level Memory Operations
    # =========================================================================

    async def remember(
        self,
        content: str,
        memory_type: MemoryUnitType = MemoryUnitType.ATOMIC_FACT,
        source: str = "agent",
        entities: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
    ) -> str:
        """
        Store a memory across appropriate systems.

        Returns the memory ID.
        """
        memory_id = None

        # Store in SimpleMem (primary long-term storage)
        if self.simple_mem:
            result = await self.simple_mem.compress(content, source)
            if result.memory_units:
                memory_id = result.memory_units[0].id

        # Store in A-MEM (for interconnected knowledge)
        if self.a_mem:
            memory = await self.a_mem.add(
                content=content,
                memory_type=memory_type,
                entities=entities,
                keywords=keywords,
                source=source,
            )
            memory_id = memory_id or memory.id

        # Store in AgeMem STM (for working memory)
        if self.age_mem:
            result = await self.age_mem.store(
                content=content,
                memory_type=memory_type,
                target="stm",
            )
            memory_id = memory_id or (result.affected_memories[0] if result.affected_memories else None)

        return memory_id or ""

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        include_episodic: bool = True,
    ) -> list[MemoryUnit]:
        """
        Recall relevant memories from all systems.

        Implements adaptive retrieval based on query complexity.
        """
        all_memories = []

        # Retrieve from SimpleMem
        if self.simple_mem:
            result = await self.simple_mem.retrieve(query)
            all_memories.extend(result.memories)

        # Retrieve from A-MEM (with graph traversal)
        if self.a_mem:
            memories = await self.a_mem.retrieve(
                query=query,
                top_k=top_k,
                include_linked=True,
            )
            all_memories.extend(memories)

        # Retrieve from AgeMem
        if self.age_mem:
            await self.age_mem.retrieve(query=query, top_k=top_k)
            all_memories.extend(self.age_mem.get_stm_contents())

        # Deduplicate by content hash
        seen_hashes = set()
        unique_memories = []
        for memory in all_memories:
            content_hash = memory.content_hash()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_memories.append(memory)

        # Sort by relevance (approximate by recency for now)
        unique_memories.sort(key=lambda m: m.created_at, reverse=True)

        return unique_memories[:top_k]

    async def get_context_for_task(
        self,
        task_description: str,
        max_tokens: int = 2000,
    ) -> str:
        """
        Get relevant context for a task (for subagent context injection).

        Implements scoped view - subagent only sees relevant memories.
        """
        # Recall relevant memories
        memories = await self.recall(task_description, top_k=10)

        # Get lessons from episodic memory
        lessons = self.episodic.get_lessons_learned(task_description)

        # Build context string
        context_parts = []

        if lessons:
            context_parts.append("Lessons from past attempts:")
            for lesson in lessons[:3]:
                context_parts.append(f"  - {lesson}")
            context_parts.append("")

        if memories:
            context_parts.append("Relevant knowledge:")
            current_tokens = 0
            for memory in memories:
                memory_tokens = len(memory.content.split()) * 4 // 3
                if current_tokens + memory_tokens > max_tokens:
                    break
                context_parts.append(f"  - {memory.content}")
                current_tokens += memory_tokens

        return "\n".join(context_parts)

    # =========================================================================
    # Artifact Management (Large Data as Handles)
    # =========================================================================

    def store_artifact(self, artifact_id: str, file_path: str) -> None:
        """Store a reference to large data (not the data itself)."""
        self.artifacts[artifact_id] = file_path

    def get_artifact_path(self, artifact_id: str) -> Optional[str]:
        """Get the file path for an artifact."""
        return self.artifacts.get(artifact_id)

    def create_artifact_reference(self, content: str, artifact_type: str) -> str:
        """
        Create an artifact for large content.

        Returns a handle string to use instead of the content.
        """
        import hashlib
        import os

        artifact_id = hashlib.md5(content.encode()).hexdigest()[:12]

        # Save to workspace
        workspace = os.path.expanduser("~/.cli-agent/artifacts")
        os.makedirs(workspace, exist_ok=True)

        file_path = os.path.join(workspace, f"{artifact_id}.{artifact_type}")
        with open(file_path, "w") as f:
            f.write(content)

        self.store_artifact(artifact_id, file_path)

        return f"[ARTIFACT:{artifact_id}:{artifact_type}]"

    # =========================================================================
    # Episodic Memory Interface
    # =========================================================================

    def start_episode(
        self,
        episode_id: str,
        task_description: str,
        attempt_number: int = 1,
    ) -> Episode:
        """Start recording a new episode."""
        return self.episodic.start_episode(
            episode_id=episode_id,
            task_description=task_description,
            attempt_number=attempt_number,
        )

    def record_action(
        self,
        action_type: str,
        action_input: Any,
        action_output: Any,
        success: bool = True,
    ) -> None:
        """Record an action in the current episode."""
        self.episodic.record_action(
            action_type=action_type,
            action_input=action_input,
            action_output=action_output,
            success=success,
        )

    async def end_episode(
        self,
        outcome: str,
        error_message: Optional[str] = None,
    ) -> Episode:
        """
        End the current episode.

        Automatically generates reflection and extracts lesson.
        """
        episode = self.episodic.get_current_episode()
        if not episode:
            raise ValueError("No episode in progress")

        # Generate reflection
        reflection = await self._generate_reflection(episode, outcome, error_message)

        # Extract lesson if failure
        lesson = None
        if outcome == "failure" and error_message:
            lesson = await self._extract_lesson(episode, error_message)

        # Store lesson in long-term memory
        if lesson:
            await self.remember(
                content=lesson,
                memory_type=MemoryUnitType.LESSON,
                source="reflexion",
            )

        return self.episodic.end_episode(
            outcome=outcome,
            error_message=error_message,
            reflection=reflection,
            lesson=lesson,
        )

    async def _generate_reflection(
        self,
        episode: Episode,
        outcome: str,
        error_message: Optional[str],
    ) -> str:
        """Generate Reflexion-style reflection on episode."""
        actions_summary = "\n".join(
            f"  - {a['type']}: {a.get('input', '')[:50]}..."
            for a in episode.actions[-5:]
        )

        prompt = f"""Reflect on this task attempt:

Task: {episode.task_description}
Outcome: {outcome}
{f'Error: {error_message}' if error_message else ''}

Recent actions:
{actions_summary}

Provide a brief reflection on what happened and what could be done differently:
"""
        return await self.llm.complete(prompt)

    async def _extract_lesson(
        self,
        episode: Episode,
        error_message: str,
    ) -> str:
        """Extract a reusable lesson from a failure."""
        prompt = f"""Extract a reusable lesson from this failure:

Task: {episode.task_description}
Error: {error_message}

The lesson should be:
1. General enough to apply to similar future situations
2. Specific enough to be actionable
3. Phrased as advice (e.g., "Always X before Y")

Lesson:
"""
        return await self.llm.complete(prompt)

    def get_reflexion_context(self, task_description: str, attempt: int) -> str:
        """Get Reflexion-style context for retrying a task."""
        return self.episodic.get_reflexion_context(task_description, attempt)

    # =========================================================================
    # Consolidation and Cleanup
    # =========================================================================

    async def consolidate(self) -> dict[str, Any]:
        """Run consolidation across all memory systems."""
        results = {}

        if self.simple_mem:
            simple_result = await self.simple_mem.consolidate()
            results["simple_mem"] = {
                "merged": simple_result.merged_count,
                "new_abstractions": simple_result.new_abstractions,
            }

        # A-MEM consolidates automatically via dynamic linking

        if self.age_mem:
            # Promote important STM to LTM
            for memory_id, slot in list(self.age_mem.stm.items()):
                if slot.importance > 0.7 or slot.access_count >= 3:
                    await self.age_mem.promote_to_ltm(memory_id)
            results["age_mem"] = {
                "stm_count": len(self.age_mem.stm),
                "ltm_count": len(self.age_mem.ltm),
            }

        return results

    async def extract_session_insights(self) -> list[str]:
        """Extract key insights from the current session."""
        # Get all recent memories
        recent_memories = []
        if self.simple_mem:
            recent_memories.extend(list(self.simple_mem.memories.values())[-20:])

        if not recent_memories:
            return []

        # Ask LLM to extract key insights
        memories_text = "\n".join(f"- {m.content}" for m in recent_memories)

        prompt = f"""Extract 3-5 key insights from these recent learnings:

{memories_text}

Each insight should be a single sentence capturing important knowledge.
Format: one insight per line.
"""
        response = await self.llm.complete(prompt)

        insights = [line.strip() for line in response.split("\n") if line.strip()]
        self.session_insights.extend(insights)

        return insights

    def stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "artifacts": len(self.artifacts),
            "session_insights": len(self.session_insights),
        }

        if self.simple_mem:
            stats["simple_mem"] = {
                "memories": len(self.simple_mem.memories),
            }

        if self.a_mem:
            stats["a_mem"] = self.a_mem.stats()

        if self.age_mem:
            stats["age_mem"] = self.age_mem.stats()

        stats["episodic"] = self.episodic.stats()

        return stats
