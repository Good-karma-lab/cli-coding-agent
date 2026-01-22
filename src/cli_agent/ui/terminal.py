"""
Terminal UI with Rich Library

Provides beautiful terminal output with:
- Syntax highlighting
- Tables and panels
- Progress bars
- Markdown rendering
"""

from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.live import Live
from rich.tree import Tree
from rich.status import Status


class RichConsole:
    """
    Rich console wrapper for CLI output.

    Provides styled output for various content types.
    """

    def __init__(
        self,
        force_terminal: bool = False,
        width: Optional[int] = None,
        record: bool = False,
    ):
        self.console = Console(
            force_terminal=force_terminal,
            width=width,
            record=record,
        )

    def print(self, *args, **kwargs) -> None:
        """Print to console."""
        self.console.print(*args, **kwargs)

    def print_code(
        self,
        code: str,
        language: str = "python",
        line_numbers: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Print syntax-highlighted code."""
        syntax = Syntax(
            code,
            language,
            line_numbers=line_numbers,
            theme="monokai",
        )
        if title:
            self.console.print(Panel(syntax, title=title))
        else:
            self.console.print(syntax)

    def print_markdown(self, markdown: str) -> None:
        """Print markdown content."""
        md = Markdown(markdown)
        self.console.print(md)

    def print_panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "blue",
    ) -> None:
        """Print content in a panel."""
        self.console.print(Panel(content, title=title, border_style=style))

    def print_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Print data as a table."""
        if not data:
            return

        table = Table(title=title)

        # Determine columns
        cols = columns or list(data[0].keys())
        for col in cols:
            table.add_column(col.replace("_", " ").title())

        # Add rows
        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in cols])

        self.console.print(table)

    def print_tree(
        self,
        data: Dict[str, Any],
        title: str = "Tree",
    ) -> None:
        """Print hierarchical data as a tree."""
        tree = Tree(title)
        self._add_tree_nodes(tree, data)
        self.console.print(tree)

    def _add_tree_nodes(self, parent: Tree, data: Dict[str, Any]) -> None:
        """Recursively add nodes to tree."""
        for key, value in data.items():
            if isinstance(value, dict):
                branch = parent.add(f"[bold]{key}[/bold]")
                self._add_tree_nodes(branch, value)
            elif isinstance(value, list):
                branch = parent.add(f"[bold]{key}[/bold] ({len(value)} items)")
                for i, item in enumerate(value[:5]):  # Show first 5
                    branch.add(str(item)[:50])
                if len(value) > 5:
                    branch.add(f"... and {len(value) - 5} more")
            else:
                parent.add(f"{key}: {value}")

    def print_error(self, message: str, title: str = "Error") -> None:
        """Print an error message."""
        self.console.print(Panel(
            f"[red]{message}[/red]",
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
        ))

    def print_success(self, message: str, title: str = "Success") -> None:
        """Print a success message."""
        self.console.print(Panel(
            f"[green]{message}[/green]",
            title=f"[bold green]{title}[/bold green]",
            border_style="green",
        ))

    def print_warning(self, message: str, title: str = "Warning") -> None:
        """Print a warning message."""
        self.console.print(Panel(
            f"[yellow]{message}[/yellow]",
            title=f"[bold yellow]{title}[/bold yellow]",
            border_style="yellow",
        ))

    def print_info(self, message: str, title: str = "Info") -> None:
        """Print an info message."""
        self.console.print(Panel(
            f"[blue]{message}[/blue]",
            title=f"[bold blue]{title}[/bold blue]",
            border_style="blue",
        ))


class TerminalUI:
    """
    Main terminal UI controller.

    Provides high-level interface for CLI interactions.
    """

    def __init__(self):
        self.console = RichConsole()
        self._progress: Optional[Progress] = None
        self._status: Optional[Status] = None

    def welcome(self, agent_name: str = "CLI Coding Agent") -> None:
        """Display welcome message."""
        self.console.print_panel(
            f"""Welcome to [bold cyan]{agent_name}[/bold cyan]

An autonomous coding assistant powered by advanced AI.

Commands:
  /help    - Show help
  /quit    - Exit
  /status  - Show agent status
  /clear   - Clear screen
            """,
            title="Welcome",
            style="cyan",
        )

    def show_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Display current task."""
        content = f"[bold]{task}[/bold]"
        if context:
            content += "\n\nContext:"
            for key, value in context.items():
                content += f"\n  {key}: {value}"

        self.console.print_panel(content, title="Current Task", style="yellow")

    def show_plan(self, steps: List[Dict[str, Any]]) -> None:
        """Display execution plan."""
        if not steps:
            return

        self.console.print("\n[bold]Execution Plan:[/bold]")

        for i, step in enumerate(steps, 1):
            status = step.get("status", "pending")
            icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌",
            }.get(status, "⬜")

            description = step.get("description", str(step))
            self.console.print(f"  {icon} {i}. {description}")

        self.console.print("")

    def show_code(
        self,
        code: str,
        language: str = "python",
        title: Optional[str] = None,
    ) -> None:
        """Display code with syntax highlighting."""
        self.console.print_code(code, language, title=title)

    def show_diff(self, diff: str) -> None:
        """Display a diff."""
        self.console.print_code(diff, "diff", line_numbers=False, title="Changes")

    def show_result(self, result: Dict[str, Any]) -> None:
        """Display task result."""
        status = result.get("status", "unknown")
        message = result.get("message", "")

        if status == "success":
            self.console.print_success(message, "Task Completed")
        elif status == "error":
            self.console.print_error(message, "Task Failed")
        else:
            self.console.print_info(message, "Result")

        # Show details if present
        if "details" in result:
            self.console.print_table(
                [result["details"]],
                title="Details",
            )

    def show_thinking(self, message: str = "Thinking...") -> Status:
        """Show thinking indicator."""
        self._status = self.console.console.status(
            f"[bold blue]{message}[/bold blue]",
            spinner="dots",
        )
        return self._status

    def start_progress(
        self,
        tasks: List[str],
    ) -> Dict[str, TaskID]:
        """Start progress tracking for multiple tasks."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console.console,
        )

        task_ids = {}
        self._progress.start()

        for task in tasks:
            task_ids[task] = self._progress.add_task(task, total=100)

        return task_ids

    def update_progress(self, task_id: TaskID, advance: float = 1) -> None:
        """Update progress for a task."""
        if self._progress:
            self._progress.update(task_id, advance=advance)

    def complete_progress(self, task_id: TaskID) -> None:
        """Mark a progress task as complete."""
        if self._progress:
            self._progress.update(task_id, completed=100)

    def stop_progress(self) -> None:
        """Stop progress tracking."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def prompt_confirm(self, message: str, default: bool = False) -> bool:
        """Prompt for yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        response = self.console.console.input(f"[bold]{message}[/bold] [{default_str}]: ")

        if not response:
            return default

        return response.lower() in ("y", "yes")

    def prompt_input(
        self,
        message: str,
        default: Optional[str] = None,
    ) -> str:
        """Prompt for text input."""
        prompt = f"[bold]{message}[/bold]"
        if default:
            prompt += f" [{default}]"
        prompt += ": "

        response = self.console.console.input(prompt)
        return response if response else (default or "")

    def prompt_choice(
        self,
        message: str,
        choices: List[str],
        default: Optional[int] = None,
    ) -> int:
        """Prompt for a choice from options."""
        self.console.print(f"\n[bold]{message}[/bold]")
        for i, choice in enumerate(choices, 1):
            marker = ">" if default == i else " "
            self.console.print(f"  {marker} {i}. {choice}")

        while True:
            response = self.console.console.input("\nChoice: ")
            if not response and default:
                return default

            try:
                choice_num = int(response)
                if 1 <= choice_num <= len(choices):
                    return choice_num
            except ValueError:
                pass

            self.console.print("[red]Invalid choice. Please try again.[/red]")

    def clear(self) -> None:
        """Clear the terminal."""
        self.console.console.clear()

    def rule(self, title: str = "") -> None:
        """Print a horizontal rule."""
        self.console.console.rule(title)
