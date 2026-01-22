"""
Interactive Prompts for User Input

Provides structured input collection with
validation and rich formatting.
"""

from typing import Any, Dict, List, Optional, Callable
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel


class InteractivePrompt:
    """
    Interactive prompt handler for CLI.

    Supports various input types with validation.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def text(
        self,
        message: str,
        default: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
        error_message: str = "Invalid input. Please try again.",
    ) -> str:
        """
        Prompt for text input.

        Args:
            message: Prompt message
            default: Default value
            validator: Optional validation function
            error_message: Error message for invalid input

        Returns:
            User input string
        """
        while True:
            result = Prompt.ask(message, console=self.console, default=default)

            if validator:
                if validator(result):
                    return result
                self.console.print(f"[red]{error_message}[/red]")
            else:
                return result

    def confirm(
        self,
        message: str,
        default: bool = False,
    ) -> bool:
        """
        Prompt for yes/no confirmation.

        Args:
            message: Prompt message
            default: Default value

        Returns:
            Boolean response
        """
        return Confirm.ask(message, console=self.console, default=default)

    def integer(
        self,
        message: str,
        default: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """
        Prompt for integer input.

        Args:
            message: Prompt message
            default: Default value
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Integer response
        """
        while True:
            result = IntPrompt.ask(message, console=self.console, default=default)

            if min_value is not None and result < min_value:
                self.console.print(f"[red]Value must be >= {min_value}[/red]")
                continue

            if max_value is not None and result > max_value:
                self.console.print(f"[red]Value must be <= {max_value}[/red]")
                continue

            return result

    def choice(
        self,
        message: str,
        choices: List[str],
        default: Optional[str] = None,
    ) -> str:
        """
        Prompt for selection from choices.

        Args:
            message: Prompt message
            choices: List of choices
            default: Default choice

        Returns:
            Selected choice
        """
        # Display choices
        self.console.print(f"\n[bold]{message}[/bold]")
        for i, choice in enumerate(choices, 1):
            marker = "*" if choice == default else " "
            self.console.print(f"  {marker} {i}. {choice}")

        while True:
            response = Prompt.ask(
                "Enter number or value",
                console=self.console,
                default=str(choices.index(default) + 1) if default else None,
            )

            # Try as number
            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
            except ValueError:
                pass

            # Try as value
            if response in choices:
                return response

            self.console.print("[red]Invalid choice. Please try again.[/red]")

    def multiselect(
        self,
        message: str,
        choices: List[str],
        defaults: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Prompt for multiple selections.

        Args:
            message: Prompt message
            choices: List of choices
            defaults: Default selections

        Returns:
            List of selected choices
        """
        defaults = defaults or []
        selected = set(defaults)

        while True:
            self.console.print(f"\n[bold]{message}[/bold]")
            self.console.print("(Enter numbers to toggle, 'done' when finished)\n")

            for i, choice in enumerate(choices, 1):
                marker = "[x]" if choice in selected else "[ ]"
                self.console.print(f"  {marker} {i}. {choice}")

            response = Prompt.ask("\nToggle", console=self.console, default="done")

            if response.lower() == "done":
                return list(selected)

            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    choice = choices[idx]
                    if choice in selected:
                        selected.remove(choice)
                    else:
                        selected.add(choice)
            except ValueError:
                self.console.print("[red]Enter a number or 'done'[/red]")

    def path(
        self,
        message: str,
        default: Optional[str] = None,
        must_exist: bool = False,
    ) -> str:
        """
        Prompt for file/directory path.

        Args:
            message: Prompt message
            default: Default path
            must_exist: Whether path must exist

        Returns:
            Path string
        """
        import os

        while True:
            result = Prompt.ask(message, console=self.console, default=default)

            if must_exist and not os.path.exists(result):
                self.console.print(f"[red]Path does not exist: {result}[/red]")
                continue

            return result

    def multiline(
        self,
        message: str,
        end_marker: str = "END",
    ) -> str:
        """
        Prompt for multiline input.

        Args:
            message: Prompt message
            end_marker: Text to end input

        Returns:
            Multiline string
        """
        self.console.print(f"[bold]{message}[/bold]")
        self.console.print(f"(Enter '{end_marker}' on a new line when done)\n")

        lines = []
        while True:
            line = self.console.input()
            if line == end_marker:
                break
            lines.append(line)

        return "\n".join(lines)

    def form(
        self,
        fields: List[Dict[str, Any]],
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prompt for multiple fields.

        Args:
            fields: List of field definitions
            title: Optional form title

        Returns:
            Dictionary of field values
        """
        if title:
            self.console.print(Panel(f"[bold]{title}[/bold]"))

        results = {}

        for field in fields:
            name = field["name"]
            field_type = field.get("type", "text")
            prompt = field.get("prompt", name)
            default = field.get("default")
            required = field.get("required", False)

            if field_type == "text":
                value = self.text(prompt, default=default)
            elif field_type == "confirm":
                value = self.confirm(prompt, default=default or False)
            elif field_type == "integer":
                value = self.integer(
                    prompt,
                    default=default,
                    min_value=field.get("min"),
                    max_value=field.get("max"),
                )
            elif field_type == "choice":
                value = self.choice(
                    prompt,
                    choices=field["choices"],
                    default=default,
                )
            elif field_type == "multiselect":
                value = self.multiselect(
                    prompt,
                    choices=field["choices"],
                    defaults=default,
                )
            elif field_type == "path":
                value = self.path(
                    prompt,
                    default=default,
                    must_exist=field.get("must_exist", False),
                )
            else:
                value = self.text(prompt, default=default)

            if required and not value:
                self.console.print("[red]This field is required[/red]")
                continue

            results[name] = value

        return results
