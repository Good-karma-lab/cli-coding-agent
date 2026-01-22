"""
Episodic Memory Implementation.

Stores sequences of actions and their outcomes for:
- Reflexion-style learning from past attempts
- Few-shot prompting with successful examples
- Experience replay for pattern recognition
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class Episode(BaseModel):
    """A single episode (sequence of actions with outcome)."""
    id: str
    task_description: str
    actions: list[dict[str, Any]]  # List of action records
    outcome: str  # success, failure, partial
    error_message: Optional[str] = None
    reflection: Optional[str] = None  # What was learned
    lesson: Optional[str] = None  # Extracted lesson for future
    created_at: datetime = datetime.utcnow()
    duration_seconds: Optional[float] = None
    tokens_used: int = 0

    # For Reflexion pattern
    attempt_number: int = 1
    previous_attempt_id: Optional[str] = None


class EpisodicMemory:
    """
    Episodic Memory for storing action sequences.

    Implements Reflexion-style learning where past episodes
    inform future attempts at similar tasks.
    """

    def __init__(
        self,
        max_episodes: int = 100,
        max_actions_per_episode: int = 50,
    ):
        self.max_episodes = max_episodes
        self.max_actions_per_episode = max_actions_per_episode

        # Episode storage
        self.episodes: dict[str, Episode] = {}
        self.episode_order: list[str] = []  # For LRU eviction

        # Index by task similarity (for few-shot retrieval)
        self.task_index: dict[str, list[str]] = {}  # keyword -> episode_ids

        # Current episode being recorded
        self._current_episode: Optional[Episode] = None

    def start_episode(
        self,
        episode_id: str,
        task_description: str,
        attempt_number: int = 1,
        previous_attempt_id: Optional[str] = None,
    ) -> Episode:
        """Start recording a new episode."""
        episode = Episode(
            id=episode_id,
            task_description=task_description,
            actions=[],
            outcome="in_progress",
            attempt_number=attempt_number,
            previous_attempt_id=previous_attempt_id,
        )
        self._current_episode = episode
        return episode

    def record_action(
        self,
        action_type: str,
        action_input: Any,
        action_output: Any,
        success: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an action in the current episode."""
        if not self._current_episode:
            return

        if len(self._current_episode.actions) >= self.max_actions_per_episode:
            return  # Episode full

        action_record = {
            "type": action_type,
            "input": str(action_input)[:1000],  # Truncate for storage
            "output": str(action_output)[:1000],
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._current_episode.actions.append(action_record)

    def end_episode(
        self,
        outcome: str,
        error_message: Optional[str] = None,
        reflection: Optional[str] = None,
        lesson: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        tokens_used: int = 0,
    ) -> Episode:
        """End the current episode and store it."""
        if not self._current_episode:
            raise ValueError("No episode in progress")

        episode = self._current_episode
        episode.outcome = outcome
        episode.error_message = error_message
        episode.reflection = reflection
        episode.lesson = lesson
        episode.duration_seconds = duration_seconds
        episode.tokens_used = tokens_used

        # Store episode
        self._store_episode(episode)
        self._current_episode = None

        return episode

    def _store_episode(self, episode: Episode) -> None:
        """Store an episode with LRU eviction."""
        # Evict if at capacity
        while len(self.episodes) >= self.max_episodes:
            oldest_id = self.episode_order.pop(0)
            self._remove_from_index(self.episodes[oldest_id])
            del self.episodes[oldest_id]

        # Store
        self.episodes[episode.id] = episode
        self.episode_order.append(episode.id)
        self._add_to_index(episode)

    def _add_to_index(self, episode: Episode) -> None:
        """Index episode by task keywords."""
        keywords = self._extract_keywords(episode.task_description)
        for keyword in keywords:
            self.task_index.setdefault(keyword, []).append(episode.id)

    def _remove_from_index(self, episode: Episode) -> None:
        """Remove episode from index."""
        keywords = self._extract_keywords(episode.task_description)
        for keyword in keywords:
            if keyword in self.task_index:
                self.task_index[keyword] = [
                    eid for eid in self.task_index[keyword]
                    if eid != episode.id
                ]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        import re
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of',
                     'and', 'or', 'in', 'for', 'on', 'with', 'at', 'by', 'from'}
        return [w for w in words if w not in stopwords and len(w) > 2]

    def get_similar_episodes(
        self,
        task_description: str,
        outcome_filter: Optional[str] = None,
        limit: int = 5,
    ) -> list[Episode]:
        """Get episodes similar to the given task."""
        keywords = self._extract_keywords(task_description)

        # Score episodes by keyword overlap
        scores: dict[str, int] = {}
        for keyword in keywords:
            for episode_id in self.task_index.get(keyword, []):
                scores[episode_id] = scores.get(episode_id, 0) + 1

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Filter and collect
        results = []
        for episode_id in sorted_ids:
            episode = self.episodes.get(episode_id)
            if not episode:
                continue
            if outcome_filter and episode.outcome != outcome_filter:
                continue
            results.append(episode)
            if len(results) >= limit:
                break

        return results

    def get_successful_examples(
        self,
        task_description: str,
        limit: int = 3,
    ) -> list[Episode]:
        """Get successful episodes for few-shot prompting."""
        return self.get_similar_episodes(
            task_description,
            outcome_filter="success",
            limit=limit,
        )

    def get_failed_attempts(
        self,
        task_description: str,
        limit: int = 3,
    ) -> list[Episode]:
        """Get failed episodes for Reflexion pattern."""
        return self.get_similar_episodes(
            task_description,
            outcome_filter="failure",
            limit=limit,
        )

    def get_lessons_learned(self, task_description: str) -> list[str]:
        """Get lessons from similar past episodes."""
        episodes = self.get_similar_episodes(task_description, limit=10)
        lessons = []
        for episode in episodes:
            if episode.lesson:
                lessons.append(episode.lesson)
        return lessons

    def get_reflexion_context(
        self,
        task_description: str,
        current_attempt: int,
    ) -> str:
        """Generate Reflexion-style context from past attempts."""
        past_failures = self.get_failed_attempts(task_description, limit=current_attempt - 1)

        if not past_failures:
            return ""

        context_parts = ["Previous attempts at similar tasks:"]

        for i, episode in enumerate(past_failures, 1):
            context_parts.append(f"\nAttempt {i}:")
            context_parts.append(f"  Task: {episode.task_description}")
            context_parts.append(f"  Outcome: {episode.outcome}")
            if episode.error_message:
                context_parts.append(f"  Error: {episode.error_message}")
            if episode.reflection:
                context_parts.append(f"  Reflection: {episode.reflection}")
            if episode.lesson:
                context_parts.append(f"  Lesson: {episode.lesson}")

        context_parts.append("\nUse these lessons to avoid making the same mistakes.")

        return "\n".join(context_parts)

    def format_for_few_shot(
        self,
        episodes: list[Episode],
        include_actions: bool = False,
    ) -> str:
        """Format episodes for few-shot prompting."""
        parts = []

        for i, episode in enumerate(episodes, 1):
            parts.append(f"Example {i}:")
            parts.append(f"  Task: {episode.task_description}")

            if include_actions and episode.actions:
                parts.append("  Actions taken:")
                for action in episode.actions[:5]:  # Limit actions shown
                    parts.append(f"    - {action['type']}: {action.get('input', '')[:100]}")

            parts.append(f"  Outcome: {episode.outcome}")
            parts.append("")

        return "\n".join(parts)

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode."""
        return self.episodes.get(episode_id)

    def get_current_episode(self) -> Optional[Episode]:
        """Get the currently recording episode."""
        return self._current_episode

    def stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        outcomes = {}
        for episode in self.episodes.values():
            outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1

        return {
            "total_episodes": len(self.episodes),
            "max_episodes": self.max_episodes,
            "outcomes": outcomes,
            "episodes_with_lessons": sum(1 for e in self.episodes.values() if e.lesson),
            "unique_keywords": len(self.task_index),
        }
