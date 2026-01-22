"""
Observability Module - OpenTelemetry, Langfuse, Loop Detection

Implements comprehensive observability from research:
- OpenTelemetry distributed tracing
- Langfuse LLM observability
- Agent loop detection
- Metrics and logging
"""

from .tracing import Tracer, Span, SpanContext
from .metrics import MetricsCollector, Counter, Histogram, Gauge
from .loop_detection import LoopDetector, LoopState, LoopAlert
from .langfuse_integration import LangfuseObserver

__all__ = [
    "Tracer",
    "Span",
    "SpanContext",
    "MetricsCollector",
    "Counter",
    "Histogram",
    "Gauge",
    "LoopDetector",
    "LoopState",
    "LoopAlert",
    "LangfuseObserver",
]
