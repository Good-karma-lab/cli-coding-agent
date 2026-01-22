"""
Distributed Tracing with OpenTelemetry

Provides tracing for agent operations with
span context propagation.
"""

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime
from enum import Enum


class SpanStatus(str, Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class SpanContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def create_root(cls) -> "SpanContext":
        """Create a root span context."""
        return cls(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
        )

    def create_child(self) -> "SpanContext":
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
        )


@dataclass
class Span:
    """
    A span representing a unit of work.

    Follows OpenTelemetry span model.
    """
    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if message and status == SpanStatus.ERROR:
            self.error = message

    def end(self, status: Optional[SpanStatus] = None) -> None:
        """End the span."""
        self.end_time = time.time()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


class Tracer:
    """
    Tracer for creating and managing spans.

    Supports OpenTelemetry-style tracing with
    context propagation.
    """

    def __init__(
        self,
        service_name: str = "cli-agent",
        exporter: Optional[Any] = None,  # OpenTelemetry exporter
    ):
        self.service_name = service_name
        self.exporter = exporter
        self._active_spans: Dict[str, Span] = {}
        self._completed_spans: List[Span] = []
        self._current_context: Optional[SpanContext] = None

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[SpanContext] = None,
    ) -> Generator[Span, None, None]:
        """
        Start a new span as context manager.

        Args:
            name: Span name
            attributes: Initial attributes
            parent_context: Optional parent context

        Yields:
            Active Span
        """
        # Create context
        if parent_context:
            context = parent_context.create_child()
        elif self._current_context:
            context = self._current_context.create_child()
        else:
            context = SpanContext.create_root()

        # Create span
        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )
        span.set_attribute("service.name", self.service_name)

        # Store as active
        self._active_spans[context.span_id] = span
        previous_context = self._current_context
        self._current_context = context

        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"message": str(e), "type": type(e).__name__})
            raise
        finally:
            span.end()
            self._current_context = previous_context
            del self._active_spans[context.span_id]
            self._completed_spans.append(span)

            # Export if exporter configured
            if self.exporter:
                self._export_span(span)

    def start_span_no_context(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a span without context manager."""
        context = (
            self._current_context.create_child()
            if self._current_context
            else SpanContext.create_root()
        )

        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )
        span.set_attribute("service.name", self.service_name)

        self._active_spans[context.span_id] = span
        return span

    def end_span(self, span: Span) -> None:
        """End a span started without context manager."""
        span.end()
        if span.context.span_id in self._active_spans:
            del self._active_spans[span.context.span_id]
        self._completed_spans.append(span)

        if self.exporter:
            self._export_span(span)

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        if self._current_context:
            return self._active_spans.get(self._current_context.span_id)
        return None

    def get_completed_spans(self) -> List[Span]:
        """Get all completed spans."""
        return self._completed_spans.copy()

    def clear_completed(self) -> None:
        """Clear completed spans."""
        self._completed_spans.clear()

    def _export_span(self, span: Span) -> None:
        """Export span to configured exporter."""
        try:
            if hasattr(self.exporter, "export"):
                self.exporter.export([span.to_dict()])
        except Exception:
            pass  # Don't fail on export errors

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a trace."""
        trace_spans = [
            s for s in self._completed_spans
            if s.context.trace_id == trace_id
        ]

        if not trace_spans:
            return {"error": "Trace not found"}

        # Build hierarchy
        root_spans = [s for s in trace_spans if s.context.parent_span_id is None]
        total_duration = sum(s.duration_ms for s in trace_spans)
        error_count = sum(1 for s in trace_spans if s.status == SpanStatus.ERROR)

        return {
            "trace_id": trace_id,
            "total_spans": len(trace_spans),
            "root_spans": len(root_spans),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "spans": [s.to_dict() for s in trace_spans],
        }
