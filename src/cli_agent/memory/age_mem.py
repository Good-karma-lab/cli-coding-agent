"""
AgeMem Implementation (arxiv:2601.01885).

RL-based unified LTM/STM management exposing memory operations
as tool-based actions:
- STORE: Add new memory
- RETRIEVE: Get relevant memories
- UPDATE: Modify existing memory
- SUMMARIZE: Compress memories
- DELETE: Remove memory
- FILTER: Remove irrelevant memories

Uses three-stage progressive RL training strategy.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from cli_agent.core.state import MemoryUnit, MemoryUnitType


class MemoryOperation(str, Enum):
    """Memory operations exposed as tools."""
    STORE = "STORE"
    RETRIEVE = "RETRIEVE"
    UPDATE = "UPDATE"
    SUMMARIZE = "SUMMARIZE"
    DELETE = "DELETE"
    FILTER = "FILTER"


class OperationResult(BaseModel):
    """Result of a memory operation."""
    operation: MemoryOperation
    success: bool
    affected_memories: list[str]  # Memory IDs
    message: str
    tokens_saved: int = 0


@dataclass
class MemorySlot:
    """A slot in short-term or long-term memory."""
    memory: MemoryUnit
    importance: float  # 0-1, used for eviction decisions
    last_access: datetime
    access_count: int = 0


class AgeMem:
    """
    AgeMem: RL-based Unified Memory Management.

    Implements LTM/STM with learnable memory operations.
    The RL policy decides WHEN and HOW to use each operation.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_client: Any,
        stm_capacity: int = 10,
        ltm_capacity: int = 1000,
        use_rl_policy: bool = False,
    ):
        self.llm = llm_client
        self.embedder = embedding_client
        self.stm_capacity = stm_capacity
        self.ltm_capacity = ltm_capacity
        self.use_rl_policy = use_rl_policy

        # Short-term memory (working memory)
        self.stm: dict[str, MemorySlot] = {}

        # Long-term memory
        self.ltm: dict[str, MemorySlot] = {}

        # Operation history for RL training
        self.operation_history: list[dict[str, Any]] = []

        # Simple importance estimator (would be learned in full RL implementation)
        self._importance_weights = {
            "recency": 0.3,
            "frequency": 0.3,
            "relevance": 0.4,
        }

    # =========================================================================
    # Memory Operations (exposed as tools to the agent)
    # =========================================================================

    async def store(
        self,
        content: str,
        memory_type: MemoryUnitType = MemoryUnitType.ATOMIC_FACT,
        target: str = "stm",  # "stm" or "ltm"
        importance: Optional[float] = None,
    ) -> OperationResult:
        """
        STORE: Add a new memory.

        By default, stores to STM. LTM storage requires explicit decision
        or automatic promotion based on importance.
        """
        # Create memory unit
        embedding = await self.embedder.embed(content)
        memory = MemoryUnit(
            type=memory_type,
            content=content,
            embedding=embedding,
            created_at=datetime.utcnow(),
        )

        # Estimate importance if not provided
        if importance is None:
            importance = await self._estimate_importance(memory)

        slot = MemorySlot(
            memory=memory,
            importance=importance,
            last_access=datetime.utcnow(),
        )

        # Store in appropriate memory
        if target == "ltm":
            await self._store_ltm(slot)
        else:
            await self._store_stm(slot)

        self._record_operation(MemoryOperation.STORE, [memory.id], True)

        return OperationResult(
            operation=MemoryOperation.STORE,
            success=True,
            affected_memories=[memory.id],
            message=f"Stored memory in {target.upper()}",
        )

    async def _store_stm(self, slot: MemorySlot) -> None:
        """Store in short-term memory, evicting if necessary."""
        # Check capacity and evict if needed
        while len(self.stm) >= self.stm_capacity:
            await self._evict_stm()

        self.stm[slot.memory.id] = slot

    async def _store_ltm(self, slot: MemorySlot) -> None:
        """Store in long-term memory, evicting if necessary."""
        while len(self.ltm) >= self.ltm_capacity:
            await self._evict_ltm()

        self.ltm[slot.memory.id] = slot

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source: str = "both",  # "stm", "ltm", or "both"
    ) -> OperationResult:
        """
        RETRIEVE: Get relevant memories based on query.

        Searches specified memory stores and returns top-k matches.
        """
        query_embedding = await self.embedder.embed(query)

        candidates = []

        # Search STM
        if source in ("stm", "both"):
            for mid, slot in self.stm.items():
                if slot.memory.embedding:
                    score = self._cosine_similarity(query_embedding, slot.memory.embedding)
                    candidates.append((score, slot, "stm"))

        # Search LTM
        if source in ("ltm", "both"):
            for mid, slot in self.ltm.items():
                if slot.memory.embedding:
                    score = self._cosine_similarity(query_embedding, slot.memory.embedding)
                    candidates.append((score, slot, "ltm"))

        # Sort and get top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:top_k]

        # Update access statistics
        retrieved_ids = []
        for _, slot, _ in top:
            slot.access_count += 1
            slot.last_access = datetime.utcnow()
            retrieved_ids.append(slot.memory.id)

        self._record_operation(MemoryOperation.RETRIEVE, retrieved_ids, True)

        return OperationResult(
            operation=MemoryOperation.RETRIEVE,
            success=True,
            affected_memories=retrieved_ids,
            message=f"Retrieved {len(retrieved_ids)} memories",
        )

    async def update(
        self,
        memory_id: str,
        new_content: Optional[str] = None,
        new_importance: Optional[float] = None,
    ) -> OperationResult:
        """
        UPDATE: Modify an existing memory.

        Can update content, importance, or both.
        """
        # Find memory
        slot = self.stm.get(memory_id) or self.ltm.get(memory_id)
        if not slot:
            return OperationResult(
                operation=MemoryOperation.UPDATE,
                success=False,
                affected_memories=[],
                message=f"Memory {memory_id} not found",
            )

        if new_content:
            slot.memory.content = new_content
            slot.memory.embedding = await self.embedder.embed(new_content)

        if new_importance is not None:
            slot.importance = new_importance

        slot.last_access = datetime.utcnow()

        self._record_operation(MemoryOperation.UPDATE, [memory_id], True)

        return OperationResult(
            operation=MemoryOperation.UPDATE,
            success=True,
            affected_memories=[memory_id],
            message="Memory updated",
        )

    async def summarize(
        self,
        memory_ids: list[str],
        target: str = "ltm",
    ) -> OperationResult:
        """
        SUMMARIZE: Compress multiple memories into one.

        Useful for consolidating related STM memories into LTM.
        """
        # Gather memories
        memories = []
        for mid in memory_ids:
            slot = self.stm.get(mid) or self.ltm.get(mid)
            if slot:
                memories.append(slot.memory)

        if len(memories) < 2:
            return OperationResult(
                operation=MemoryOperation.SUMMARIZE,
                success=False,
                affected_memories=[],
                message="Need at least 2 memories to summarize",
            )

        # Generate summary
        contents = [m.content for m in memories]
        prompt = f"""Summarize these memories into a single coherent memory:

{chr(10).join(f"- {c}" for c in contents)}

Keep important details but remove redundancy. Output only the summary:
"""
        summary = await self.llm.complete(prompt)

        # Create summarized memory
        embedding = await self.embedder.embed(summary.strip())
        summarized = MemoryUnit(
            type=MemoryUnitType.INSIGHT,
            content=summary.strip(),
            embedding=embedding,
            source_memories=memory_ids,
            abstraction_level=1,
        )

        # Calculate average importance
        avg_importance = sum(
            (self.stm.get(mid) or self.ltm.get(mid)).importance
            for mid in memory_ids
            if (self.stm.get(mid) or self.ltm.get(mid))
        ) / len(memory_ids)

        # Store summarized memory
        slot = MemorySlot(
            memory=summarized,
            importance=avg_importance,
            last_access=datetime.utcnow(),
        )

        if target == "ltm":
            await self._store_ltm(slot)
        else:
            await self._store_stm(slot)

        # Calculate tokens saved
        original_tokens = sum(self._estimate_tokens(m.content) for m in memories)
        new_tokens = self._estimate_tokens(summary.strip())
        tokens_saved = original_tokens - new_tokens

        self._record_operation(MemoryOperation.SUMMARIZE, [summarized.id], True)

        return OperationResult(
            operation=MemoryOperation.SUMMARIZE,
            success=True,
            affected_memories=[summarized.id],
            message=f"Summarized {len(memories)} memories",
            tokens_saved=tokens_saved,
        )

    async def delete(self, memory_id: str) -> OperationResult:
        """
        DELETE: Remove a specific memory.
        """
        deleted = False
        if memory_id in self.stm:
            del self.stm[memory_id]
            deleted = True
        if memory_id in self.ltm:
            del self.ltm[memory_id]
            deleted = True

        self._record_operation(MemoryOperation.DELETE, [memory_id], deleted)

        return OperationResult(
            operation=MemoryOperation.DELETE,
            success=deleted,
            affected_memories=[memory_id] if deleted else [],
            message="Memory deleted" if deleted else "Memory not found",
        )

    async def filter(
        self,
        query: str,
        threshold: float = 0.3,
        source: str = "stm",
    ) -> OperationResult:
        """
        FILTER: Remove memories below relevance threshold.

        Cleans up irrelevant memories to save context space.
        """
        query_embedding = await self.embedder.embed(query)
        storage = self.stm if source == "stm" else self.ltm

        to_remove = []
        for mid, slot in storage.items():
            if slot.memory.embedding:
                score = self._cosine_similarity(query_embedding, slot.memory.embedding)
                if score < threshold:
                    to_remove.append(mid)

        for mid in to_remove:
            del storage[mid]

        # Calculate tokens saved
        tokens_saved = sum(
            self._estimate_tokens(storage.get(mid, MemorySlot(
                memory=MemoryUnit(content=""),
                importance=0,
                last_access=datetime.utcnow()
            )).memory.content)
            for mid in to_remove
        )

        self._record_operation(MemoryOperation.FILTER, to_remove, True)

        return OperationResult(
            operation=MemoryOperation.FILTER,
            success=True,
            affected_memories=to_remove,
            message=f"Filtered {len(to_remove)} memories",
            tokens_saved=tokens_saved,
        )

    # =========================================================================
    # Automatic Memory Management
    # =========================================================================

    async def _evict_stm(self) -> None:
        """Evict least important STM memory (potentially promoting to LTM)."""
        if not self.stm:
            return

        # Find lowest importance
        min_importance = float('inf')
        evict_id = None

        for mid, slot in self.stm.items():
            if slot.importance < min_importance:
                min_importance = slot.importance
                evict_id = mid

        if evict_id:
            slot = self.stm.pop(evict_id)

            # Potentially promote to LTM if high access count
            if slot.access_count >= 3:
                await self._store_ltm(slot)

    async def _evict_ltm(self) -> None:
        """Evict least important LTM memory."""
        if not self.ltm:
            return

        # Combined score: importance * recency * frequency
        min_score = float('inf')
        evict_id = None

        for mid, slot in self.ltm.items():
            age_hours = (datetime.utcnow() - slot.last_access).total_seconds() / 3600
            recency_score = 1 / (1 + age_hours / 24)
            frequency_score = min(slot.access_count / 10, 1)

            combined = (
                self._importance_weights["relevance"] * slot.importance +
                self._importance_weights["recency"] * recency_score +
                self._importance_weights["frequency"] * frequency_score
            )

            if combined < min_score:
                min_score = combined
                evict_id = mid

        if evict_id:
            del self.ltm[evict_id]

    async def _estimate_importance(self, memory: MemoryUnit) -> float:
        """Estimate memory importance (would be learned in RL implementation)."""
        # Simple heuristics
        content = memory.content.lower()

        importance = 0.5  # Base importance

        # Boost for specific content types
        if any(kw in content for kw in ['error', 'bug', 'fix', 'important', 'critical']):
            importance += 0.2
        if any(kw in content for kw in ['decision', 'chose', 'selected', 'because']):
            importance += 0.15
        if any(kw in content for kw in ['user', 'request', 'require', 'must']):
            importance += 0.1

        # Length-based adjustment
        words = len(content.split())
        if words > 50:
            importance += 0.1
        elif words < 10:
            importance -= 0.1

        return min(max(importance, 0), 1)

    async def promote_to_ltm(self, memory_id: str) -> bool:
        """Explicitly promote a STM memory to LTM."""
        if memory_id not in self.stm:
            return False

        slot = self.stm.pop(memory_id)
        await self._store_ltm(slot)
        return True

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _record_operation(
        self,
        operation: MemoryOperation,
        affected: list[str],
        success: bool,
    ) -> None:
        """Record operation for RL training."""
        self.operation_history.append({
            "operation": operation.value,
            "affected_memories": affected,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "stm_size": len(self.stm),
            "ltm_size": len(self.ltm),
        })

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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text.split()) * 4 // 3

    def get_stm_contents(self) -> list[MemoryUnit]:
        """Get all STM memories."""
        return [slot.memory for slot in self.stm.values()]

    def get_ltm_contents(self) -> list[MemoryUnit]:
        """Get all LTM memories."""
        return [slot.memory for slot in self.ltm.values()]

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "stm_count": len(self.stm),
            "stm_capacity": self.stm_capacity,
            "ltm_count": len(self.ltm),
            "ltm_capacity": self.ltm_capacity,
            "total_operations": len(self.operation_history),
            "stm_avg_importance": sum(s.importance for s in self.stm.values()) / max(len(self.stm), 1),
            "ltm_avg_importance": sum(s.importance for s in self.ltm.values()) / max(len(self.ltm), 1),
        }
