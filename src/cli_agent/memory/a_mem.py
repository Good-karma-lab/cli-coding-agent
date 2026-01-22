"""
A-MEM Implementation (NeurIPS 2025).

Zettelkasten-inspired interconnected knowledge networks.
Achieves superior performance using only ~2,000 tokens vs MemGPT's ~16,900.

Key features:
- Dynamic linking based on semantic similarities
- Memory evolution where new memories trigger updates to existing ones
- Bidirectional linking for graph traversal
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from cli_agent.core.state import MemoryUnit, MemoryUnitType


class LinkUpdate(BaseModel):
    """Record of a dynamic link update."""
    source_id: str
    target_id: str
    old_strength: float
    new_strength: float
    reason: str
    timestamp: datetime


class AMem:
    """
    A-MEM: Zettelkasten-inspired Agentic Memory.

    Implements interconnected knowledge networks with:
    - Atomic notes (memory units)
    - Bidirectional links with strength scores
    - Dynamic link updates when new memories arrive
    - Graph-based retrieval
    """

    def __init__(
        self,
        embedding_client: Any,
        link_threshold: float = 0.7,
        max_memories: int = 1000,
        enable_dynamic_linking: bool = True,
    ):
        self.embedder = embedding_client
        self.link_threshold = link_threshold
        self.max_memories = max_memories
        self.enable_dynamic_linking = enable_dynamic_linking

        # Memory storage
        self.memories: dict[str, MemoryUnit] = {}

        # Link graph: memory_id -> {linked_id: strength}
        self.links: dict[str, dict[str, float]] = {}

        # Update history for transparency
        self.link_updates: list[LinkUpdate] = []

        # Index structures for efficient retrieval
        self._keyword_index: dict[str, set[str]] = {}  # keyword -> memory_ids
        self._entity_index: dict[str, set[str]] = {}  # entity -> memory_ids

    async def add(
        self,
        content: str,
        memory_type: MemoryUnitType = MemoryUnitType.ATOMIC_FACT,
        entities: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        source: str = "user",
    ) -> MemoryUnit:
        """
        Add a new memory to the Zettelkasten.

        This triggers:
        1. Embedding generation
        2. Automatic linking to related memories
        3. Dynamic updates to existing memories if relevant
        """
        # Enforce capacity limit (LRU eviction)
        if len(self.memories) >= self.max_memories:
            self._evict_least_accessed()

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Create memory unit
        memory = MemoryUnit(
            type=memory_type,
            content=content,
            entities=entities or [],
            keywords=keywords or [],
            embedding=embedding,
            source=source,
            created_at=datetime.utcnow(),
        )

        # Store memory
        self.memories[memory.id] = memory
        self.links[memory.id] = {}

        # Update indices
        self._index_memory(memory)

        # Find and create links to related memories
        await self._create_links(memory)

        # Dynamic linking: update existing memories if this adds new context
        if self.enable_dynamic_linking:
            await self._trigger_dynamic_updates(memory)

        return memory

    async def _create_links(self, memory: MemoryUnit) -> None:
        """Create bidirectional links to semantically related memories."""
        if not memory.embedding:
            return

        for other_id, other in self.memories.items():
            if other_id == memory.id:
                continue

            if not other.embedding:
                continue

            # Calculate semantic similarity
            similarity = self._cosine_similarity(memory.embedding, other.embedding)

            if similarity >= self.link_threshold:
                # Create bidirectional link
                self.links[memory.id][other_id] = similarity
                self.links.setdefault(other_id, {})[memory.id] = similarity

                # Update memory objects
                memory.linked_memories.append(other_id)
                memory.link_strengths[other_id] = similarity
                other.linked_memories.append(memory.id)
                other.link_strengths[memory.id] = similarity

    async def _trigger_dynamic_updates(self, new_memory: MemoryUnit) -> None:
        """
        Dynamic linking: new memories can trigger updates to existing ones.

        For example, if we learn new information about an entity, we might
        want to strengthen links to other memories about that entity.
        """
        # Find memories that share entities
        shared_entity_memories = set()
        for entity in new_memory.entities:
            if entity in self._entity_index:
                shared_entity_memories.update(self._entity_index[entity])

        # Remove self
        shared_entity_memories.discard(new_memory.id)

        for other_id in shared_entity_memories:
            other = self.memories.get(other_id)
            if not other:
                continue

            # Check if we should strengthen or create a link
            current_strength = self.links.get(new_memory.id, {}).get(other_id, 0)

            # Boost for shared entities
            shared_entities = set(new_memory.entities) & set(other.entities)
            entity_boost = min(0.1 * len(shared_entities), 0.3)

            new_strength = min(current_strength + entity_boost, 1.0)

            if new_strength > current_strength and new_strength >= self.link_threshold:
                # Record the update
                self.link_updates.append(LinkUpdate(
                    source_id=new_memory.id,
                    target_id=other_id,
                    old_strength=current_strength,
                    new_strength=new_strength,
                    reason=f"Shared entities: {shared_entities}",
                    timestamp=datetime.utcnow(),
                ))

                # Update links
                self.links.setdefault(new_memory.id, {})[other_id] = new_strength
                self.links.setdefault(other_id, {})[new_memory.id] = new_strength

    def _index_memory(self, memory: MemoryUnit) -> None:
        """Add memory to keyword and entity indices."""
        for keyword in memory.keywords:
            self._keyword_index.setdefault(keyword.lower(), set()).add(memory.id)

        for entity in memory.entities:
            self._entity_index.setdefault(entity.lower(), set()).add(memory.id)

    def _evict_least_accessed(self) -> None:
        """Evict the least recently accessed memory."""
        if not self.memories:
            return

        # Find least accessed memory
        min_access = float('inf')
        evict_id = None

        for mid, memory in self.memories.items():
            access_score = memory.access_count
            if memory.last_accessed:
                # Penalize old memories
                age_hours = (datetime.utcnow() - memory.last_accessed).total_seconds() / 3600
                access_score = access_score / (1 + age_hours / 24)

            if access_score < min_access:
                min_access = access_score
                evict_id = mid

        if evict_id:
            self._remove_memory(evict_id)

    def _remove_memory(self, memory_id: str) -> None:
        """Remove a memory and all its links."""
        memory = self.memories.get(memory_id)
        if not memory:
            return

        # Remove from indices
        for keyword in memory.keywords:
            if keyword.lower() in self._keyword_index:
                self._keyword_index[keyword.lower()].discard(memory_id)

        for entity in memory.entities:
            if entity.lower() in self._entity_index:
                self._entity_index[entity.lower()].discard(memory_id)

        # Remove links
        if memory_id in self.links:
            for linked_id in self.links[memory_id]:
                if linked_id in self.links:
                    self.links[linked_id].pop(memory_id, None)
            del self.links[memory_id]

        # Remove memory
        del self.memories[memory_id]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_linked: bool = True,
        max_hops: int = 2,
    ) -> list[MemoryUnit]:
        """
        Retrieve relevant memories using graph-based search.

        1. Find top-k semantically similar memories
        2. Optionally expand to linked memories (graph traversal)
        """
        # Get query embedding
        query_embedding = await self.embedder.embed(query)

        # Score all memories by similarity
        scored = []
        for mid, memory in self.memories.items():
            if memory.embedding:
                score = self._cosine_similarity(query_embedding, memory.embedding)
                scored.append((score, mid))

        # Sort and get top-k
        scored.sort(reverse=True)
        top_ids = [mid for _, mid in scored[:top_k]]

        # Optionally expand via links
        if include_linked:
            expanded = set(top_ids)
            frontier = set(top_ids)

            for _ in range(max_hops):
                new_frontier = set()
                for mid in frontier:
                    linked = self.links.get(mid, {})
                    for linked_id, strength in linked.items():
                        if linked_id not in expanded and strength >= self.link_threshold:
                            new_frontier.add(linked_id)
                            expanded.add(linked_id)

                frontier = new_frontier
                if not frontier:
                    break

            top_ids = list(expanded)

        # Get memories and update access stats
        results = []
        for mid in top_ids:
            memory = self.memories.get(mid)
            if memory:
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                results.append(memory)

        # Sort by relevance (original score)
        if results:
            results.sort(
                key=lambda m: self._cosine_similarity(query_embedding, m.embedding or []),
                reverse=True
            )

        return results

    def get_connected_component(self, memory_id: str) -> list[MemoryUnit]:
        """Get all memories in the same connected component."""
        if memory_id not in self.memories:
            return []

        visited = set()
        queue = [memory_id]
        component = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)
            memory = self.memories.get(current)
            if memory:
                component.append(memory)

            # Add linked memories to queue
            for linked_id in self.links.get(current, {}):
                if linked_id not in visited:
                    queue.append(linked_id)

        return component

    def get_link_graph(self) -> dict[str, list[tuple[str, float]]]:
        """Get the link graph for visualization."""
        graph = {}
        for source_id, links in self.links.items():
            graph[source_id] = [(target, strength) for target, strength in links.items()]
        return graph

    def search_by_entity(self, entity: str) -> list[MemoryUnit]:
        """Search memories by entity."""
        memory_ids = self._entity_index.get(entity.lower(), set())
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]

    def search_by_keyword(self, keyword: str) -> list[MemoryUnit]:
        """Search memories by keyword."""
        memory_ids = self._keyword_index.get(keyword.lower(), set())
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        total_links = sum(len(links) for links in self.links.values()) // 2  # Bidirectional
        avg_links = total_links / max(len(self.memories), 1)

        return {
            "total_memories": len(self.memories),
            "total_links": total_links,
            "avg_links_per_memory": avg_links,
            "link_updates": len(self.link_updates),
            "unique_entities": len(self._entity_index),
            "unique_keywords": len(self._keyword_index),
        }
