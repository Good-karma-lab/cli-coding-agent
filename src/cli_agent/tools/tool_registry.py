"""
Tool Registry - Central registry for all tools.

Manages tool registration, discovery, and execution.
Provides unified interface for agent tool use.
"""

from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass, field
from pydantic import BaseModel
import inspect


@dataclass
class Tool:
    """
    Tool definition for the registry.

    Wraps tool classes with metadata for discovery
    and execution.
    """
    name: str
    description: str
    instance: Any
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None
    categories: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    is_async: bool = False

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        # Create input if schema exists
        if self.input_schema:
            input_data = self.input_schema(**kwargs)
            return self.instance.execute(input_data)
        else:
            return self.instance.execute(**kwargs)

    async def execute_async(self, **kwargs) -> Any:
        """Execute the tool asynchronously."""
        if self.input_schema:
            input_data = self.input_schema(**kwargs)
            if hasattr(self.instance, "execute_async"):
                return await self.instance.execute_async(input_data)
            else:
                return self.instance.execute(input_data)
        else:
            if hasattr(self.instance, "execute_async"):
                return await self.instance.execute_async(**kwargs)
            else:
                return self.instance.execute(**kwargs)

    def get_schema_for_llm(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        schema = {
            "name": self.name,
            "description": self.description,
        }

        if self.input_schema:
            # Convert Pydantic model to JSON schema
            schema["parameters"] = self.input_schema.model_json_schema()
        else:
            schema["parameters"] = {"type": "object", "properties": {}}

        return schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "categories": self.categories,
            "requires_confirmation": self.requires_confirmation,
            "is_async": self.is_async,
            "has_schema": self.input_schema is not None,
        }


class ToolRegistry:
    """
    Central registry for all tools.

    Provides:
    - Tool registration and discovery
    - Category-based organization
    - Unified execution interface
    - LLM-compatible schema generation
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._confirmation_required: set = set()

    def register(
        self,
        tool_class: Any,
        categories: Optional[List[str]] = None,
        requires_confirmation: bool = False,
        **kwargs,
    ) -> Tool:
        """
        Register a tool.

        Args:
            tool_class: Tool class to instantiate
            categories: Categories for organization
            requires_confirmation: Whether tool needs user confirmation
            **kwargs: Arguments for tool instantiation

        Returns:
            Registered Tool instance
        """
        # Instantiate tool
        instance = tool_class(**kwargs)

        # Get metadata from tool class
        name = getattr(instance, "name", tool_class.__name__)
        description = getattr(instance, "description", tool_class.__doc__ or "")
        input_schema = getattr(instance, "input_schema", None)
        output_schema = getattr(instance, "output_schema", None)

        # Check if async
        is_async = hasattr(instance, "execute_async")

        # Create tool wrapper
        tool = Tool(
            name=name,
            description=description,
            instance=instance,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=categories or [],
            requires_confirmation=requires_confirmation,
            is_async=is_async,
        )

        # Register
        self._tools[name] = tool

        # Index by category
        for category in (categories or []):
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)

        if requires_confirmation:
            self._confirmation_required.add(name)

        return tool

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        categories: Optional[List[str]] = None,
        requires_confirmation: bool = False,
    ) -> Tool:
        """
        Register a function as a tool.

        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            input_schema: Pydantic model for input validation
            categories: Categories for organization
            requires_confirmation: Whether tool needs user confirmation

        Returns:
            Registered Tool instance
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""

        # Create wrapper class
        class FunctionTool:
            def execute(self, input_data=None, **kwargs):
                if input_data:
                    return func(**input_data.model_dump())
                return func(**kwargs)

        instance = FunctionTool()
        instance.name = tool_name
        instance.description = tool_description
        instance.input_schema = input_schema
        instance.output_schema = None

        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            async def execute_async(self, input_data=None, **kwargs):
                if input_data:
                    return await func(**input_data.model_dump())
                return await func(**kwargs)
            instance.execute_async = execute_async.__get__(instance, FunctionTool)

        tool = Tool(
            name=tool_name,
            description=tool_description,
            instance=instance,
            input_schema=input_schema,
            categories=categories or [],
            requires_confirmation=requires_confirmation,
            is_async=is_async,
        )

        self._tools[tool_name] = tool

        for category in (categories or []):
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool_name)

        if requires_confirmation:
            self._confirmation_required.add(tool_name)

        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._categories.keys())

    def execute(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        return tool.execute(**kwargs)

    async def execute_async(self, name: str, **kwargs) -> Any:
        """
        Execute a tool asynchronously.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        return await tool.execute_async(**kwargs)

    def requires_confirmation(self, name: str) -> bool:
        """Check if tool requires user confirmation."""
        return name in self._confirmation_required

    def get_schemas_for_llm(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for LLM function calling."""
        return [tool.get_schema_for_llm() for tool in self._tools.values()]

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool definitions in OpenAI function format."""
        tools = []
        for tool in self._tools.values():
            schema = tool.get_schema_for_llm()
            tools.append({
                "type": "function",
                "function": schema,
            })
        return tools

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.

        Returns True if tool was found and removed.
        """
        if name not in self._tools:
            return False

        tool = self._tools[name]

        # Remove from categories
        for category in tool.categories:
            if category in self._categories:
                self._categories[category] = [
                    n for n in self._categories[category] if n != name
                ]

        # Remove from confirmation set
        self._confirmation_required.discard(name)

        # Remove tool
        del self._tools[name]

        return True

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()
        self._confirmation_required.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        return {
            "tools": [tool.to_dict() for tool in self._tools.values()],
            "categories": self._categories,
            "total_tools": len(self._tools),
        }


def create_default_registry(workspace_root: Optional[str] = None) -> ToolRegistry:
    """
    Create a registry with default tools.

    Args:
        workspace_root: Workspace root path

    Returns:
        ToolRegistry with default tools registered
    """
    from .file_tools import (
        FileReadTool, FileWriteTool, FileEditTool,
        FileSearchTool, DirectoryListTool,
    )
    from .git_tools import (
        GitStatusTool, GitDiffTool, GitCommitTool,
        GitBranchTool, GitLogTool, GitWorktreeTool,
    )
    from .shell_tools import ShellExecuteTool, SafeShellExecutor
    from .code_tools import CodeSearchTool, SymbolLookupTool, ReferenceFinderTool

    registry = ToolRegistry()

    # File tools
    registry.register(
        FileReadTool,
        categories=["file", "read"],
        workspace_root=workspace_root,
    )
    registry.register(
        FileWriteTool,
        categories=["file", "write"],
        requires_confirmation=True,
        workspace_root=workspace_root,
    )
    registry.register(
        FileEditTool,
        categories=["file", "write"],
        requires_confirmation=True,
        workspace_root=workspace_root,
    )
    registry.register(
        FileSearchTool,
        categories=["file", "search"],
        workspace_root=workspace_root,
    )
    registry.register(
        DirectoryListTool,
        categories=["file", "read"],
        workspace_root=workspace_root,
    )

    # Git tools
    registry.register(
        GitStatusTool,
        categories=["git", "read"],
    )
    registry.register(
        GitDiffTool,
        categories=["git", "read"],
    )
    registry.register(
        GitCommitTool,
        categories=["git", "write"],
        requires_confirmation=True,
    )
    registry.register(
        GitBranchTool,
        categories=["git"],
    )
    registry.register(
        GitLogTool,
        categories=["git", "read"],
    )
    registry.register(
        GitWorktreeTool,
        categories=["git", "workspace"],
    )

    # Shell tools
    executor = SafeShellExecutor(workspace_root)
    registry.register(
        ShellExecuteTool,
        categories=["shell", "execute"],
        requires_confirmation=True,
        executor=executor,
    )

    # Code tools
    registry.register(
        CodeSearchTool,
        categories=["code", "search"],
        workspace_root=workspace_root,
    )
    registry.register(
        SymbolLookupTool,
        categories=["code", "navigation"],
        workspace_root=workspace_root,
    )
    registry.register(
        ReferenceFinderTool,
        categories=["code", "navigation"],
        workspace_root=workspace_root,
    )

    return registry
