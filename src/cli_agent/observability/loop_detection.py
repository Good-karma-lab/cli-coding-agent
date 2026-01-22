"""
Agent Loop Detection

Detects when agents get stuck in loops or repetitive patterns.
Critical for preventing infinite loops and wasted resources.

Based on research for agent observability and safety.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
from collections import deque


class LoopSeverity(str, Enum):
    """Severity of detected loop."""
    INFO = "info"          # Minor repetition
    WARNING = "warning"    # Potential issue
    CRITICAL = "critical"  # Definite loop


class LoopType(str, Enum):
    """Type of loop detected."""
    ACTION_REPEAT = "action_repeat"      # Same action repeated
    STATE_CYCLE = "state_cycle"          # State returns to previous
    OUTPUT_REPEAT = "output_repeat"      # Same output generated
    ERROR_CYCLE = "error_cycle"          # Same error repeating
    OSCILLATION = "oscillation"          # Alternating between states


@dataclass
class LoopState:
    """State tracked for loop detection."""
    action_history: List[str] = field(default_factory=list)
    state_history: List[str] = field(default_factory=list)
    output_history: List[str] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Detection thresholds
    max_history: int = 100
    action_repeat_threshold: int = 3
    state_cycle_threshold: int = 2
    error_repeat_threshold: int = 3

    def add_action(self, action: str) -> None:
        """Record an action."""
        self.action_history.append(action)
        self.timestamps.append(time.time())
        self._trim_history()

    def add_state(self, state_hash: str) -> None:
        """Record a state hash."""
        self.state_history.append(state_hash)
        self._trim_history()

    def add_output(self, output_hash: str) -> None:
        """Record an output hash."""
        self.output_history.append(output_hash)
        self._trim_history()

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.error_history.append(error)
        self._trim_history()

    def _trim_history(self) -> None:
        """Keep history bounded."""
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        if len(self.output_history) > self.max_history:
            self.output_history = self.output_history[-self.max_history:]
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        if len(self.timestamps) > self.max_history:
            self.timestamps = self.timestamps[-self.max_history:]

    def clear(self) -> None:
        """Clear all history."""
        self.action_history.clear()
        self.state_history.clear()
        self.output_history.clear()
        self.error_history.clear()
        self.timestamps.clear()


@dataclass
class LoopAlert:
    """Alert generated when loop is detected."""
    loop_type: LoopType
    severity: LoopSeverity
    message: str
    pattern: List[str]
    repeat_count: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.loop_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "pattern": self.pattern,
            "repeat_count": self.repeat_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class LoopDetector:
    """
    Detects loops and repetitive patterns in agent behavior.

    Monitors:
    - Action repetition
    - State cycles
    - Output repetition
    - Error cycles
    - Oscillating behavior
    """

    def __init__(
        self,
        action_repeat_threshold: int = 3,
        state_cycle_threshold: int = 2,
        output_repeat_threshold: int = 3,
        error_repeat_threshold: int = 3,
        window_size: int = 20,
        on_alert: Optional[Callable[[LoopAlert], None]] = None,
    ):
        self.action_repeat_threshold = action_repeat_threshold
        self.state_cycle_threshold = state_cycle_threshold
        self.output_repeat_threshold = output_repeat_threshold
        self.error_repeat_threshold = error_repeat_threshold
        self.window_size = window_size
        self.on_alert = on_alert

        self._state = LoopState(
            action_repeat_threshold=action_repeat_threshold,
            state_cycle_threshold=state_cycle_threshold,
            error_repeat_threshold=error_repeat_threshold,
        )
        self._alerts: List[LoopAlert] = []
        self._suppressed_patterns: Set[str] = set()

    def record_action(self, action: str) -> Optional[LoopAlert]:
        """
        Record an action and check for loops.

        Args:
            action: Action description or identifier

        Returns:
            LoopAlert if loop detected
        """
        self._state.add_action(action)
        return self._check_action_loops()

    def record_state(self, state: Dict[str, Any]) -> Optional[LoopAlert]:
        """
        Record agent state and check for cycles.

        Args:
            state: Agent state dictionary

        Returns:
            LoopAlert if cycle detected
        """
        state_hash = self._hash_state(state)
        self._state.add_state(state_hash)
        return self._check_state_cycles()

    def record_output(self, output: str) -> Optional[LoopAlert]:
        """
        Record agent output and check for repetition.

        Args:
            output: Agent output

        Returns:
            LoopAlert if repetition detected
        """
        output_hash = self._hash_output(output)
        self._state.add_output(output_hash)
        return self._check_output_repetition()

    def record_error(self, error: str) -> Optional[LoopAlert]:
        """
        Record an error and check for error cycles.

        Args:
            error: Error message

        Returns:
            LoopAlert if error cycle detected
        """
        self._state.add_error(error)
        return self._check_error_cycles()

    def check_all(
        self,
        action: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> List[LoopAlert]:
        """
        Record multiple signals and check for all loop types.

        Returns:
            List of LoopAlerts detected
        """
        alerts = []

        if action:
            alert = self.record_action(action)
            if alert:
                alerts.append(alert)

        if state:
            alert = self.record_state(state)
            if alert:
                alerts.append(alert)

        if output:
            alert = self.record_output(output)
            if alert:
                alerts.append(alert)

        if error:
            alert = self.record_error(error)
            if alert:
                alerts.append(alert)

        # Check for oscillation
        osc_alert = self._check_oscillation()
        if osc_alert:
            alerts.append(osc_alert)

        return alerts

    def _check_action_loops(self) -> Optional[LoopAlert]:
        """Check for repeated actions."""
        history = self._state.action_history
        if len(history) < self.action_repeat_threshold:
            return None

        recent = history[-self.window_size:]

        # Check for simple repetition
        last_action = recent[-1]
        repeat_count = 0
        for action in reversed(recent):
            if action == last_action:
                repeat_count += 1
            else:
                break

        if repeat_count >= self.action_repeat_threshold:
            return self._create_alert(
                LoopType.ACTION_REPEAT,
                LoopSeverity.WARNING,
                f"Action '{last_action}' repeated {repeat_count} times",
                [last_action],
                repeat_count,
            )

        # Check for pattern repetition (e.g., A-B-A-B)
        pattern_alert = self._detect_pattern(recent)
        if pattern_alert:
            return pattern_alert

        return None

    def _check_state_cycles(self) -> Optional[LoopAlert]:
        """Check for state cycles."""
        history = self._state.state_history
        if len(history) < self.state_cycle_threshold * 2:
            return None

        recent = history[-self.window_size:]

        # Count occurrences of each state
        state_counts: Dict[str, int] = {}
        for state_hash in recent:
            state_counts[state_hash] = state_counts.get(state_hash, 0) + 1

        # Find repeated states
        for state_hash, count in state_counts.items():
            if count >= self.state_cycle_threshold:
                return self._create_alert(
                    LoopType.STATE_CYCLE,
                    LoopSeverity.WARNING,
                    f"State repeated {count} times in window",
                    [state_hash],
                    count,
                )

        return None

    def _check_output_repetition(self) -> Optional[LoopAlert]:
        """Check for repeated outputs."""
        history = self._state.output_history
        if len(history) < self.output_repeat_threshold:
            return None

        recent = history[-self.window_size:]

        # Count output occurrences
        output_counts: Dict[str, int] = {}
        for output_hash in recent:
            output_counts[output_hash] = output_counts.get(output_hash, 0) + 1

        for output_hash, count in output_counts.items():
            if count >= self.output_repeat_threshold:
                return self._create_alert(
                    LoopType.OUTPUT_REPEAT,
                    LoopSeverity.WARNING,
                    f"Output repeated {count} times",
                    [output_hash],
                    count,
                )

        return None

    def _check_error_cycles(self) -> Optional[LoopAlert]:
        """Check for repeated errors."""
        history = self._state.error_history
        if len(history) < self.error_repeat_threshold:
            return None

        recent = history[-self.window_size:]

        # Count error occurrences
        error_counts: Dict[str, int] = {}
        for error in recent:
            error_counts[error] = error_counts.get(error, 0) + 1

        for error, count in error_counts.items():
            if count >= self.error_repeat_threshold:
                return self._create_alert(
                    LoopType.ERROR_CYCLE,
                    LoopSeverity.CRITICAL,
                    f"Error '{error[:50]}...' repeated {count} times",
                    [error],
                    count,
                )

        return None

    def _check_oscillation(self) -> Optional[LoopAlert]:
        """Check for oscillating behavior (A-B-A-B pattern)."""
        history = self._state.action_history
        if len(history) < 4:
            return None

        recent = history[-self.window_size:]

        # Look for A-B-A-B pattern
        if len(recent) >= 4:
            # Check last 4 elements
            a, b, c, d = recent[-4:]
            if a == c and b == d and a != b:
                # Found oscillation pattern
                return self._create_alert(
                    LoopType.OSCILLATION,
                    LoopSeverity.WARNING,
                    f"Oscillating between '{a}' and '{b}'",
                    [a, b],
                    2,
                )

        return None

    def _detect_pattern(self, sequence: List[str]) -> Optional[LoopAlert]:
        """Detect repeating patterns in sequence."""
        n = len(sequence)

        # Try different pattern lengths
        for pattern_len in range(2, n // 2 + 1):
            pattern = sequence[-pattern_len:]
            repeat_count = 0

            # Check how many times pattern repeats
            for i in range(n - pattern_len, -1, -pattern_len):
                window = sequence[i:i + pattern_len]
                if window == pattern:
                    repeat_count += 1
                else:
                    break

            if repeat_count >= 2:
                return self._create_alert(
                    LoopType.ACTION_REPEAT,
                    LoopSeverity.WARNING,
                    f"Pattern of length {pattern_len} repeated {repeat_count} times",
                    pattern,
                    repeat_count,
                )

        return None

    def _create_alert(
        self,
        loop_type: LoopType,
        severity: LoopSeverity,
        message: str,
        pattern: List[str],
        repeat_count: int,
    ) -> Optional[LoopAlert]:
        """Create and register an alert."""
        # Check if pattern is suppressed
        pattern_key = f"{loop_type.value}:{str(pattern)}"
        if pattern_key in self._suppressed_patterns:
            return None

        alert = LoopAlert(
            loop_type=loop_type,
            severity=severity,
            message=message,
            pattern=pattern,
            repeat_count=repeat_count,
        )

        self._alerts.append(alert)

        # Call handler if registered
        if self.on_alert:
            self.on_alert(alert)

        return alert

    def suppress_pattern(self, loop_type: LoopType, pattern: List[str]) -> None:
        """Suppress alerts for a specific pattern."""
        pattern_key = f"{loop_type.value}:{str(pattern)}"
        self._suppressed_patterns.add(pattern_key)

    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create hash of state dictionary."""
        state_str = str(sorted(state.items()))
        return hashlib.md5(state_str.encode()).hexdigest()[:16]

    def _hash_output(self, output: str) -> str:
        """Create hash of output."""
        return hashlib.md5(output.encode()).hexdigest()[:16]

    def get_alerts(self, severity: Optional[LoopSeverity] = None) -> List[LoopAlert]:
        """Get alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self._alerts if a.severity == severity]
        return self._alerts.copy()

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()

    def reset(self) -> None:
        """Reset all state and alerts."""
        self._state.clear()
        self._alerts.clear()
        self._suppressed_patterns.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of detector state."""
        return {
            "total_actions": len(self._state.action_history),
            "total_states": len(self._state.state_history),
            "total_outputs": len(self._state.output_history),
            "total_errors": len(self._state.error_history),
            "total_alerts": len(self._alerts),
            "critical_alerts": len([a for a in self._alerts if a.severity == LoopSeverity.CRITICAL]),
            "suppressed_patterns": len(self._suppressed_patterns),
        }
