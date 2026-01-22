"""
Metrics Collection for Agent Monitoring

Provides counters, histograms, and gauges for
tracking agent performance.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict
import threading


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    Counter metric that only increases.

    Use for counting events like requests, errors, etc.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._labels_key(labels)
        return self._values.get(key, 0.0)

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._values.clear()

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Create key from labels."""
        if not labels:
            return ""
        return str(sorted(labels.items()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": "counter",
            "description": self.description,
            "values": dict(self._values),
        }


class Histogram:
    """
    Histogram metric for measuring distributions.

    Use for measuring latencies, sizes, etc.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._values: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a value."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key].append(value)

    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get statistics for the histogram."""
        key = self._labels_key(labels)
        values = self._values.get(key, [])

        if not values:
            return {
                "count": 0,
                "sum": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "p50": 0,
                "p90": 0,
                "p99": 0,
            }

        sorted_values = sorted(values)
        count = len(values)

        def percentile(p: float) -> float:
            idx = int(count * p)
            return sorted_values[min(idx, count - 1)]

        return {
            "count": count,
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / count,
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "p99": percentile(0.99),
        }

    def get_bucket_counts(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, int]:
        """Get bucket counts."""
        key = self._labels_key(labels)
        values = self._values.get(key, [])

        counts = {}
        for bucket in self.buckets:
            counts[f"le_{bucket}"] = sum(1 for v in values if v <= bucket)
        counts["le_inf"] = len(values)

        return counts

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._values.clear()

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return str(sorted(labels.items()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": "histogram",
            "description": self.description,
            "buckets": self.buckets,
            "stats": {k: self.get_stats({"key": k}) for k in self._values},
        }


class Gauge:
    """
    Gauge metric that can increase or decrease.

    Use for measuring current values like queue size, memory, etc.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._labels_key(labels)
        return self._values.get(key, 0.0)

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return str(sorted(labels.items()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": "gauge",
            "description": self.description,
            "values": dict(self._values),
        }


class MetricsCollector:
    """
    Central metrics collector and registry.

    Manages all metrics and provides export functionality.
    """

    def __init__(self, prefix: str = "cli_agent"):
        self.prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._gauges: Dict[str, Gauge] = {}

        # Create default metrics
        self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """Initialize default metrics."""
        # Counters
        self.create_counter("requests_total", "Total requests processed")
        self.create_counter("errors_total", "Total errors encountered")
        self.create_counter("llm_calls_total", "Total LLM API calls")
        self.create_counter("tool_calls_total", "Total tool invocations")

        # Histograms
        self.create_histogram("request_duration_seconds", "Request duration")
        self.create_histogram("llm_latency_seconds", "LLM call latency")
        self.create_histogram("tool_latency_seconds", "Tool execution latency")
        self.create_histogram("tokens_used", "Tokens used per request",
                              buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000])

        # Gauges
        self.create_gauge("active_tasks", "Number of active tasks")
        self.create_gauge("memory_usage_bytes", "Memory usage")
        self.create_gauge("pending_messages", "Pending messages in queue")

    def create_counter(self, name: str, description: str = "") -> Counter:
        """Create a counter metric."""
        full_name = f"{self.prefix}_{name}"
        counter = Counter(full_name, description)
        self._counters[name] = counter
        return counter

    def create_histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Create a histogram metric."""
        full_name = f"{self.prefix}_{name}"
        histogram = Histogram(full_name, description, buckets)
        self._histograms[name] = histogram
        return histogram

    def create_gauge(self, name: str, description: str = "") -> Gauge:
        """Create a gauge metric."""
        full_name = f"{self.prefix}_{name}"
        gauge = Gauge(full_name, description)
        self._gauges[name] = gauge
        return gauge

    def counter(self, name: str) -> Optional[Counter]:
        """Get a counter by name."""
        return self._counters.get(name)

    def histogram(self, name: str) -> Optional[Histogram]:
        """Get a histogram by name."""
        return self._histograms.get(name)

    def gauge(self, name: str) -> Optional[Gauge]:
        """Get a gauge by name."""
        return self._gauges.get(name)

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        counter = self._counters.get(name)
        if counter:
            counter.inc(value, labels)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        histogram = self._histograms.get(name)
        if histogram:
            histogram.observe(value, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        gauge = self._gauges.get(name)
        if gauge:
            gauge.set(value, labels)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        return {
            "counters": {n: c.to_dict() for n, c in self._counters.items()},
            "histograms": {n: h.to_dict() for n, h in self._histograms.items()},
            "gauges": {n: g.to_dict() for n, g in self._gauges.items()},
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, counter in self._counters.items():
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            for labels_key, value in counter._values.items():
                labels_str = f"{{{labels_key}}}" if labels_key else ""
                lines.append(f"{counter.name}{labels_str} {value}")

        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {histogram.name} {histogram.description}")
            lines.append(f"# TYPE {histogram.name} histogram")
            for labels_key in histogram._values:
                stats = histogram.get_stats({"key": labels_key} if labels_key else None)
                labels_str = f"{{{labels_key}}}" if labels_key else ""
                lines.append(f"{histogram.name}_count{labels_str} {stats['count']}")
                lines.append(f"{histogram.name}_sum{labels_str} {stats['sum']}")

        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for labels_key, value in gauge._values.items():
                labels_str = f"{{{labels_key}}}" if labels_key else ""
                lines.append(f"{gauge.name}{labels_str} {value}")

        return "\n".join(lines)

    def reset_all(self) -> None:
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.reset()
        for histogram in self._histograms.values():
            histogram.reset()
        for gauge in self._gauges.values():
            gauge._values.clear()
