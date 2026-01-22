"""
Display Components for CLI

Provides progress bars, status displays, and
real-time updates.
"""

from typing import Any, Dict, List, Optional, Callable
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn, TaskID,
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
import time
import threading


class ProgressDisplay:
    """
    Progress tracking display.

    Shows progress for multiple concurrent tasks.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._progress: Optional[Progress] = None
        self._tasks: Dict[str, TaskID] = {}

    def start(self, tasks: Optional[List[str]] = None) -> None:
        """
        Start progress display.

        Args:
            tasks: Initial tasks to track
        """
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._progress.start()

        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(
        self,
        description: str,
        total: float = 100.0,
    ) -> str:
        """
        Add a task to track.

        Args:
            description: Task description
            total: Total units for completion

        Returns:
            Task ID string
        """
        if not self._progress:
            self.start()

        task_id = self._progress.add_task(description, total=total)
        self._tasks[description] = task_id
        return description

    def update(
        self,
        task: str,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Update task progress.

        Args:
            task: Task ID or description
            advance: Amount to advance
            completed: Set completed amount
            description: Update description
        """
        if not self._progress:
            return

        task_id = self._tasks.get(task)
        if task_id is None:
            return

        kwargs = {}
        if advance is not None:
            kwargs["advance"] = advance
        if completed is not None:
            kwargs["completed"] = completed
        if description is not None:
            kwargs["description"] = description

        self._progress.update(task_id, **kwargs)

    def complete(self, task: str) -> None:
        """Mark task as complete."""
        if not self._progress:
            return

        task_id = self._tasks.get(task)
        if task_id is not None:
            self._progress.update(task_id, completed=100)

    def stop(self) -> None:
        """Stop progress display."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self._tasks.clear()


class StatusDisplay:
    """
    Real-time status display.

    Shows updating status information in panels.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._live: Optional[Live] = None
        self._status: Dict[str, Any] = {}
        self._layout: Optional[Layout] = None

    def start(self, sections: Optional[List[str]] = None) -> None:
        """
        Start status display.

        Args:
            sections: Section names to display
        """
        sections = sections or ["Status"]
        self._layout = Layout()

        # Create sections
        self._layout.split_column(
            *[Layout(name=s) for s in sections]
        )

        self._live = Live(
            self._layout,
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def update_section(
        self,
        section: str,
        content: Any,
        title: Optional[str] = None,
    ) -> None:
        """
        Update a status section.

        Args:
            section: Section name
            content: Content to display
            title: Optional panel title
        """
        if not self._layout:
            return

        if isinstance(content, dict):
            # Format as table
            table = Table(show_header=False, box=None)
            for key, value in content.items():
                table.add_row(f"[bold]{key}[/bold]", str(value))
            display = Panel(table, title=title or section)
        elif isinstance(content, list):
            # Format as list
            items = "\n".join(f"• {item}" for item in content)
            display = Panel(items, title=title or section)
        else:
            display = Panel(str(content), title=title or section)

        try:
            self._layout[section].update(display)
        except KeyError:
            pass

    def set_status(self, key: str, value: Any) -> None:
        """Set a status value."""
        self._status[key] = value

        if self._layout:
            self.update_section("Status", self._status)

    def stop(self) -> None:
        """Stop status display."""
        if self._live:
            self._live.stop()
            self._live = None
            self._layout = None


class StreamingOutput:
    """
    Streaming output display.

    Shows streaming text with optional syntax highlighting.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._buffer = ""
        self._live: Optional[Live] = None

    def start(self, title: Optional[str] = None) -> None:
        """Start streaming display."""
        self._buffer = ""
        self._live = Live(
            Panel(self._buffer, title=title or "Output"),
            console=self.console,
            refresh_per_second=10,
        )
        self._live.start()

    def write(self, text: str) -> None:
        """
        Write text to stream.

        Args:
            text: Text to append
        """
        self._buffer += text

        if self._live:
            self._live.update(Panel(self._buffer[-2000:]))  # Keep last 2000 chars

    def writeline(self, text: str) -> None:
        """Write line to stream."""
        self.write(text + "\n")

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = ""
        if self._live:
            self._live.update(Panel(""))

    def stop(self) -> str:
        """
        Stop streaming and return content.

        Returns:
            Complete buffer content
        """
        if self._live:
            self._live.stop()
            self._live = None

        return self._buffer

    def get_content(self) -> str:
        """Get current buffer content."""
        return self._buffer


class AgentDisplay:
    """
    Combined display for agent operations.

    Shows plan, progress, and status together.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = ProgressDisplay(console)
        self.status = StatusDisplay(console)
        self._streaming: Optional[StreamingOutput] = None

    def show_agent_panel(
        self,
        task: str,
        status: str = "Starting",
        plan: Optional[List[str]] = None,
    ) -> None:
        """
        Show initial agent panel.

        Args:
            task: Current task
            status: Current status
            plan: Execution plan steps
        """
        content = f"[bold]Task:[/bold] {task}\n"
        content += f"[bold]Status:[/bold] {status}\n"

        if plan:
            content += "\n[bold]Plan:[/bold]\n"
            for i, step in enumerate(plan, 1):
                content += f"  {i}. {step}\n"

        self.console.print(Panel(content, title="Agent", border_style="blue"))

    def start_execution(self, steps: List[str]) -> None:
        """
        Start execution progress.

        Args:
            steps: Steps to execute
        """
        self.progress.start(steps)

    def update_step(
        self,
        step: str,
        progress: float,
        message: Optional[str] = None,
    ) -> None:
        """
        Update step progress.

        Args:
            step: Step name
            progress: Progress percentage
            message: Optional status message
        """
        self.progress.update(step, completed=progress, description=message)

    def complete_step(self, step: str) -> None:
        """Mark step as complete."""
        self.progress.complete(step)

    def start_streaming(self, title: str = "Output") -> None:
        """Start streaming output."""
        self._streaming = StreamingOutput(self.console)
        self._streaming.start(title)

    def stream_text(self, text: str) -> None:
        """Stream text output."""
        if self._streaming:
            self._streaming.write(text)

    def stop_streaming(self) -> str:
        """Stop streaming and return content."""
        if self._streaming:
            content = self._streaming.stop()
            self._streaming = None
            return content
        return ""

    def stop(self) -> None:
        """Stop all displays."""
        self.progress.stop()
        self.status.stop()
        if self._streaming:
            self._streaming.stop()
            self._streaming = None
