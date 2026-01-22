"""
Langfuse Integration for LLM Observability

Provides observability for LLM calls including:
- Token usage tracking
- Latency monitoring
- Cost tracking
- Generation quality metrics
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime


@dataclass
class LLMCall:
    """Record of a single LLM call."""
    id: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Generation:
    """Record of a generation (prompt + completion)."""
    id: str
    name: str
    trace_id: Optional[str] = None
    model: str = ""
    prompt: str = ""
    completion: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class LangfuseObserver:
    """
    Observer for LLM calls with Langfuse-style tracking.

    Features:
    - Token usage tracking
    - Latency monitoring
    - Cost estimation
    - Generation recording
    """

    # Pricing per 1K tokens (approximate)
    MODEL_PRICING = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    }

    def __init__(
        self,
        langfuse_client: Optional[Any] = None,  # Actual Langfuse client
        track_prompts: bool = True,
        track_completions: bool = True,
    ):
        self.langfuse_client = langfuse_client
        self.track_prompts = track_prompts
        self.track_completions = track_completions

        self._calls: List[LLMCall] = []
        self._generations: List[Generation] = []
        self._total_tokens = 0
        self._total_cost = 0.0
        self._call_counter = 0

    def _generate_id(self) -> str:
        """Generate unique ID."""
        self._call_counter += 1
        return f"call_{self._call_counter}_{int(time.time())}"

    def start_generation(
        self,
        name: str,
        model: str,
        prompt: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generation:
        """
        Start tracking a generation.

        Args:
            name: Generation name
            model: Model being used
            prompt: Input prompt
            trace_id: Optional trace ID
            metadata: Optional metadata

        Returns:
            Generation record
        """
        gen = Generation(
            id=self._generate_id(),
            name=name,
            trace_id=trace_id,
            model=model,
            prompt=prompt if self.track_prompts else "[not tracked]",
            metadata=metadata or {},
        )

        # If using Langfuse client, create generation
        if self.langfuse_client:
            try:
                self.langfuse_client.generation(
                    name=name,
                    model=model,
                    input=prompt,
                    metadata=metadata,
                )
            except Exception:
                pass

        return gen

    def end_generation(
        self,
        generation: Generation,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
    ) -> None:
        """
        Complete a generation record.

        Args:
            generation: Generation to complete
            completion: Generated completion
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            latency_ms: Latency in milliseconds
        """
        generation.completion = completion if self.track_completions else "[not tracked]"
        generation.prompt_tokens = prompt_tokens
        generation.completion_tokens = completion_tokens
        generation.latency_ms = latency_ms

        self._generations.append(generation)

        # Update totals
        total_tokens = prompt_tokens + completion_tokens
        self._total_tokens += total_tokens

        # Calculate cost
        cost = self._calculate_cost(
            generation.model,
            prompt_tokens,
            completion_tokens,
        )
        self._total_cost += cost

        # Record LLM call
        self._calls.append(LLMCall(
            id=generation.id,
            model=generation.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost=cost,
            metadata=generation.metadata,
        ))

        # Update Langfuse if available
        if self.langfuse_client:
            try:
                self.langfuse_client.generation(
                    name=generation.name,
                    model=generation.model,
                    output=completion,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    metadata=generation.metadata,
                )
            except Exception:
                pass

    def record_error(
        self,
        generation: Generation,
        error: str,
        latency_ms: float,
    ) -> None:
        """
        Record a failed generation.

        Args:
            generation: Generation that failed
            error: Error message
            latency_ms: Latency before failure
        """
        generation.latency_ms = latency_ms

        self._calls.append(LLMCall(
            id=generation.id,
            model=generation.model,
            latency_ms=latency_ms,
            success=False,
            error=error,
            metadata=generation.metadata,
        ))

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for tokens."""
        # Find pricing
        pricing = None
        for model_key, prices in self.MODEL_PRICING.items():
            if model_key in model.lower():
                pricing = prices
                break

        if not pricing:
            return 0.0

        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        if not self._calls:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }

        latencies = [c.latency_ms for c in self._calls if c.success]
        errors = sum(1 for c in self._calls if not c.success)

        return {
            "total_calls": len(self._calls),
            "successful_calls": len(self._calls) - errors,
            "failed_calls": errors,
            "total_tokens": self._total_tokens,
            "total_cost": round(self._total_cost, 4),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "tokens_per_call": self._total_tokens / len(self._calls) if self._calls else 0,
        }

    def get_model_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics broken down by model."""
        breakdown: Dict[str, Dict[str, Any]] = {}

        for call in self._calls:
            model = call.model
            if model not in breakdown:
                breakdown[model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "errors": 0,
                    "latencies": [],
                }

            breakdown[model]["calls"] += 1
            breakdown[model]["tokens"] += call.total_tokens
            breakdown[model]["cost"] += call.cost
            if not call.success:
                breakdown[model]["errors"] += 1
            else:
                breakdown[model]["latencies"].append(call.latency_ms)

        # Calculate averages
        for model, stats in breakdown.items():
            latencies = stats.pop("latencies")
            stats["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0
            stats["cost"] = round(stats["cost"], 4)

        return breakdown

    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent LLM calls."""
        recent = self._calls[-limit:]
        return [
            {
                "id": c.id,
                "model": c.model,
                "tokens": c.total_tokens,
                "latency_ms": c.latency_ms,
                "cost": round(c.cost, 6),
                "success": c.success,
                "timestamp": c.timestamp.isoformat(),
            }
            for c in recent
        ]

    def get_generations(
        self,
        limit: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> List[Generation]:
        """Get generation records."""
        gens = self._generations

        if trace_id:
            gens = [g for g in gens if g.trace_id == trace_id]

        if limit:
            gens = gens[-limit:]

        return gens

    def flush(self) -> None:
        """Flush to Langfuse if client configured."""
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
            except Exception:
                pass

    def reset(self) -> None:
        """Reset all tracking."""
        self._calls.clear()
        self._generations.clear()
        self._total_tokens = 0
        self._total_cost = 0.0
        self._call_counter = 0
