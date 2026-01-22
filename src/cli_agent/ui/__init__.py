"""
UI Module - CLI Interface with Rich Terminal

Implements terminal UI from research:
- Rich terminal formatting
- Interactive prompts
- Progress displays
- Syntax highlighting
"""

from .terminal import TerminalUI, RichConsole
from .prompts import InteractivePrompt
from .displays import ProgressDisplay, StatusDisplay

__all__ = [
    "TerminalUI",
    "RichConsole",
    "InteractivePrompt",
    "ProgressDisplay",
    "StatusDisplay",
]
