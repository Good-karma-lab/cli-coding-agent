"""
SimpleMem Implementation (arxiv:2601.02553).

Three-stage pipeline for efficient lifelong memory:
1. Semantic Structured Compression
2. Recursive Memory Consolidation
3. Adaptive Query-Aware Retrieval

Achieves 26.4% F1 improvement with 30x token reduction.
"""

import asyncio
import hashlib
import re
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from cli_agent.core.state import MemoryUnit, MemoryUnitType


class CompressionResult(BaseModel):
    """Result of Stage 1 compression."""
    memory_units: list[MemoryUnit]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float


class ConsolidationResult(BaseModel):
    """Result of Stage 2 consolidation."""
    consolidated_units: list[MemoryUnit]
    merged_count: int
    new_abstractions: int


class RetrievalResult(BaseModel):
    """Result of Stage 3 retrieval."""
    memories: list[MemoryUnit]
    query_complexity: str  # low, medium, high
    retrieval_scope: int
    tokens_used: int


class SimpleMem:
    """
    SimpleMem: Efficient Lifelong Memory for LLM Agents.

    Implements Semantic Lossless Compression to achieve ~550 tokens
    with 43.24% F1 score - optimal performance-efficiency frontier.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_client: Any,
        entropy_threshold: float = 0.3,
        similarity_threshold: float = 0.85,
        min_retrieval: int = 3,
        max_retrieval: int = 20,
    ):
        self.llm = llm_client
        self.embedder = embedding_client
        self.entropy_threshold = entropy_threshold
        self.similarity_threshold = similarity_threshold
        self.min_retrieval = min_retrieval
        self.max_retrieval = max_retrieval

        # Memory storage
        self.memories: dict[str, MemoryUnit] = {}
        self.embeddings_cache: dict[str, list[float]] = {}

        # Consolidation queue
        self._consolidation_queue: list[str] = []
        self._consolidation_interval = 10

    # =========================================================================
    # Stage 1: Semantic Structured Compression
    # =========================================================================

    async def compress(
        self,
        raw_content: str,
        source: str = "conversation",
        timestamp: Optional[datetime] = None,
    ) -> CompressionResult:
        """
        Stage 1: Semantic Structured Compression.

        Transforms raw content into compact, multi-view indexed memory units:
        - Entropy-aware filtering (remove low-information content)
        - Coreference resolution (pronouns -> actual names)
        - Temporal anchoring (relative -> absolute timestamps)
        """
        timestamp = timestamp or datetime.utcnow()
        original_tokens = self._estimate_tokens(raw_content)

        # Step 1: Extract atomic facts with coreference resolution
        atomic_facts = await self._extract_atomic_facts(raw_content)

        # Step 2: Filter by entropy (remove low-information content)
        filtered_facts = self._filter_by_entropy(atomic_facts)

        # Step 3: Apply temporal anchoring
        anchored_facts = self._anchor_temporally(filtered_facts, timestamp)

        # Step 4: Create memory units with multi-view indexing
        memory_units = []
        for fact in anchored_facts:
            unit = await self._create_memory_unit(fact, source, timestamp)
            memory_units.append(unit)
            self.memories[unit.id] = unit
            self._consolidation_queue.append(unit.id)

        # Trigger consolidation if queue is full
        if len(self._consolidation_queue) >= self._consolidation_interval:
            asyncio.create_task(self._background_consolidation())

        compressed_tokens = sum(self._estimate_tokens(u.content) for u in memory_units)

        return CompressionResult(
            memory_units=memory_units,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=original_tokens / max(compressed_tokens, 1),
        )

    async def _extract_atomic_facts(self, content: str) -> list[dict[str, Any]]:
        """Extract independent, self-contained facts using LLM."""
        prompt = f"""Extract atomic facts from the following content.

Rules:
1. Each fact must be self-contained (no pronouns - use actual names)
2. Each fact should be independent (understandable without context)
3. Preserve important details but remove redundancy
4. Convert relative times to descriptions (e.g., "earlier" -> "before this interaction")

Content:
{content}

Output each fact on a new line, prefixed with "FACT: "
Also identify entities, prefixed with "ENTITIES: " (comma-separated)
"""
        response = await self.llm.complete(prompt)

        facts = []
        current_fact = None
        current_entities = []

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('FACT:'):
                if current_fact:
                    facts.append({'content': current_fact, 'entities': current_entities})
                current_fact = line[5:].strip()
                current_entities = []
            elif line.startswith('ENTITIES:'):
                current_entities = [e.strip() for e in line[9:].split(',') if e.strip()]

        if current_fact:
            facts.append({'content': current_fact, 'entities': current_entities})

        return facts

    def _filter_by_entropy(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out low-entropy (low-information) facts."""
        filtered = []
        for fact in facts:
            content = fact['content']

            # Simple entropy estimation based on:
            # - Length (very short facts often lack substance)
            # - Unique words ratio
            # - Presence of specific details (numbers, names, etc.)

            words = content.split()
            if len(words) < 3:
                continue

            unique_ratio = len(set(words)) / len(words)

            # Check for specificity indicators
            has_specifics = bool(
                re.search(r'\d+', content) or  # Numbers
                re.search(r'[A-Z][a-z]+', content) or  # Proper nouns
                any(kw in content.lower() for kw in ['because', 'therefore', 'since', 'when', 'where', 'how'])
            )

            entropy_score = unique_ratio * (1.5 if has_specifics else 1.0)

            if entropy_score >= self.entropy_threshold:
                filtered.append(fact)

        return filtered

    def _anchor_temporally(
        self,
        facts: list[dict[str, Any]],
        reference_time: datetime,
    ) -> list[dict[str, Any]]:
        """Convert relative temporal references to absolute."""
        temporal_patterns = [
            (r'\byesterday\b', (reference_time.replace(hour=0, minute=0, second=0)).isoformat()[:10]),
            (r'\btoday\b', reference_time.isoformat()[:10]),
            (r'\bnow\b', reference_time.isoformat()),
            (r'\bearlier\b', f'before {reference_time.isoformat()}'),
            (r'\blater\b', f'after {reference_time.isoformat()}'),
            (r'\bjust now\b', reference_time.isoformat()),
        ]

        anchored = []
        for fact in facts:
            content = fact['content']
            for pattern, replacement in temporal_patterns:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            anchored.append({**fact, 'content': content})

        return anchored

    async def _create_memory_unit(
        self,
        fact: dict[str, Any],
        source: str,
        timestamp: datetime,
    ) -> MemoryUnit:
        """Create a memory unit with multi-view indexing."""
        content = fact['content']
        entities = fact.get('entities', [])

        # Generate embedding
        embedding = await self.embedder.embed(content)

        # Extract keywords
        keywords = self._extract_keywords(content)

        return MemoryUnit(
            type=MemoryUnitType.ATOMIC_FACT,
            content=content,
            entities=entities,
            created_at=timestamp,
            keywords=keywords,
            embedding=embedding,
            source=source,
            abstraction_level=0,
        )

    def _extract_keywords(self, content: str) -> list[str]:
        """Extract keywords from content."""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        # Filter common words and keep meaningful ones
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'that', 'this', 'these', 'those', 'it', 'its'}
        keywords = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
        # Return unique keywords, preserving order
        seen = set()
        return [k for k in keywords if not (k in seen or seen.add(k))][:10]

    # =========================================================================
    # Stage 2: Recursive Memory Consolidation
    # =========================================================================

    async def consolidate(self) -> ConsolidationResult:
        """
        Stage 2: Recursive Memory Consolidation.

        Asynchronously integrates related memory units into higher-level
        abstract representations, reducing redundancy while preserving
        semantic content.
        """
        to_consolidate = [self.memories[mid] for mid in self._consolidation_queue if mid in self.memories]
        self._consolidation_queue.clear()

        if len(to_consolidate) < 2:
            return ConsolidationResult(
                consolidated_units=[],
                merged_count=0,
                new_abstractions=0,
            )

        # Find similar memory clusters
        clusters = await self._cluster_similar_memories(to_consolidate)

        merged_count = 0
        new_abstractions = 0
        consolidated = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Merge cluster into higher-level abstraction
            abstract_unit = await self._merge_cluster(cluster)
            if abstract_unit:
                self.memories[abstract_unit.id] = abstract_unit
                consolidated.append(abstract_unit)
                new_abstractions += 1
                merged_count += len(cluster)

        return ConsolidationResult(
            consolidated_units=consolidated,
            merged_count=merged_count,
            new_abstractions=new_abstractions,
        )

    async def _cluster_similar_memories(
        self,
        memories: list[MemoryUnit],
    ) -> list[list[MemoryUnit]]:
        """Cluster memories by semantic similarity."""
        if not memories:
            return []

        # Build similarity matrix
        n = len(memories)
        clusters: list[list[MemoryUnit]] = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            cluster = [memories[i]]
            used.add(i)

            for j in range(i + 1, n):
                if j in used:
                    continue

                sim = self._cosine_similarity(
                    memories[i].embedding or [],
                    memories[j].embedding or []
                )

                if sim >= self.similarity_threshold:
                    cluster.append(memories[j])
                    used.add(j)

            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    async def _merge_cluster(self, cluster: list[MemoryUnit]) -> Optional[MemoryUnit]:
        """Merge a cluster of similar memories into an abstraction."""
        contents = [m.content for m in cluster]
        all_entities = list(set(e for m in cluster for e in m.entities))
        all_keywords = list(set(k for m in cluster for k in m.keywords))

        prompt = f"""Synthesize these related facts into a single, higher-level insight:

Facts:
{chr(10).join(f"- {c}" for c in contents)}

Rules:
1. Preserve all important information
2. Remove redundancy
3. Create a more abstract, general statement if appropriate
4. Keep it self-contained (no pronouns)

Output the synthesized insight:
"""
        response = await self.llm.complete(prompt)
        synthesized = response.strip()

        if not synthesized:
            return None

        # Generate new embedding
        embedding = await self.embedder.embed(synthesized)

        return MemoryUnit(
            type=MemoryUnitType.INSIGHT,
            content=synthesized,
            entities=all_entities[:10],
            keywords=all_keywords[:10],
            embedding=embedding,
            abstraction_level=max(m.abstraction_level for m in cluster) + 1,
            source_memories=[m.id for m in cluster],
            source="consolidation",
        )

    async def _background_consolidation(self) -> None:
        """Run consolidation in background."""
        try:
            await self.consolidate()
        except Exception:
            pass  # Don't fail on background consolidation errors

    # =========================================================================
    # Stage 3: Adaptive Query-Aware Retrieval
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        max_tokens: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Stage 3: Adaptive Query-Aware Retrieval.

        Dynamically adjusts retrieval scope based on query complexity,
        constructing precise context efficiently.
        """
        # Assess query complexity
        complexity = await self._assess_query_complexity(query)

        # Determine retrieval scope based on complexity
        if complexity == "low":
            scope = self.min_retrieval
        elif complexity == "high":
            scope = self.max_retrieval
        else:
            scope = (self.min_retrieval + self.max_retrieval) // 2

        # Get query embedding
        query_embedding = await self.embedder.embed(query)

        # Retrieve most relevant memories
        scored_memories = []
        for memory in self.memories.values():
            if memory.embedding:
                score = self._cosine_similarity(query_embedding, memory.embedding)
                scored_memories.append((score, memory))

        # Sort by relevance and take top N
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        retrieved = [m for _, m in scored_memories[:scope]]

        # Update access statistics
        for memory in retrieved:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()

        # Calculate tokens used
        tokens_used = sum(self._estimate_tokens(m.content) for m in retrieved)

        return RetrievalResult(
            memories=retrieved,
            query_complexity=complexity,
            retrieval_scope=scope,
            tokens_used=tokens_used,
        )

    async def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity to determine retrieval scope."""
        # Simple heuristics first
        words = query.split()

        # Low complexity: simple, short queries
        if len(words) < 5:
            return "low"

        # High complexity indicators
        high_complexity_indicators = [
            'compare', 'contrast', 'relationship', 'between', 'all', 'every',
            'history', 'evolution', 'throughout', 'across', 'comprehensive',
            'analyze', 'evaluate', 'explain why', 'how does'
        ]

        query_lower = query.lower()
        if any(ind in query_lower for ind in high_complexity_indicators):
            return "high"

        # Count question words
        question_words = sum(1 for w in ['what', 'why', 'how', 'when', 'where', 'which', 'who']
                           if w in query_lower)
        if question_words > 1:
            return "high"

        return "medium"

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 4 // 3  # ~1.33 tokens per word average

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def get_all_memories(self) -> list[MemoryUnit]:
        """Get all stored memories."""
        return list(self.memories.values())

    def get_memory(self, memory_id: str) -> Optional[MemoryUnit]:
        """Get a specific memory by ID."""
        return self.memories.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self._consolidation_queue.clear()
