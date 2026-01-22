"""
File Tools - Deterministic file operations with Pydantic validation.

All file tools are:
- Type-safe with Pydantic schemas
- Deterministic (same input = same output)
- Safe with path validation
- Well-documented for LLM tool use
"""

import os
import glob as glob_module
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import fnmatch


class FileReadInput(BaseModel):
    """Input schema for file read operation."""
    path: str = Field(description="Path to the file to read")
    start_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="Starting line number (1-indexed)"
    )
    end_line: Optional[int] = Field(
        default=None,
        ge=1,
        description="Ending line number (1-indexed, inclusive)"
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path is not empty and doesn't traverse up."""
        if not v:
            raise ValueError("Path cannot be empty")
        if ".." in v:
            raise ValueError("Path cannot contain '..'")
        return v


class FileReadOutput(BaseModel):
    """Output schema for file read operation."""
    success: bool
    content: Optional[str] = None
    lines: Optional[List[str]] = None
    total_lines: int = 0
    error: Optional[str] = None
    path: str = ""


class FileReadTool:
    """
    Read file contents.

    Supports reading entire files or specific line ranges.
    Returns content with line numbers for easy reference.
    """

    name: str = "file_read"
    description: str = "Read contents of a file, optionally specifying line range"
    input_schema = FileReadInput
    output_schema = FileReadOutput

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute(self, input_data: FileReadInput) -> FileReadOutput:
        """Execute file read operation."""
        try:
            # Resolve path
            full_path = self._resolve_path(input_data.path)

            if not os.path.exists(full_path):
                return FileReadOutput(
                    success=False,
                    error=f"File not found: {input_data.path}",
                    path=input_data.path,
                )

            if os.path.isdir(full_path):
                return FileReadOutput(
                    success=False,
                    error=f"Path is a directory: {input_data.path}",
                    path=input_data.path,
                )

            # Read file
            with open(full_path, "r", encoding=input_data.encoding) as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Apply line range
            start = (input_data.start_line or 1) - 1
            end = input_data.end_line or total_lines

            selected_lines = all_lines[start:end]
            content = "".join(selected_lines)

            return FileReadOutput(
                success=True,
                content=content,
                lines=selected_lines,
                total_lines=total_lines,
                path=input_data.path,
            )

        except UnicodeDecodeError as e:
            return FileReadOutput(
                success=False,
                error=f"Encoding error: {str(e)}",
                path=input_data.path,
            )
        except Exception as e:
            return FileReadOutput(
                success=False,
                error=str(e),
                path=input_data.path,
            )

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)


class FileWriteInput(BaseModel):
    """Input schema for file write operation."""
    path: str = Field(description="Path to write file")
    content: str = Field(description="Content to write")
    create_dirs: bool = Field(
        default=True,
        description="Create parent directories if needed"
    )
    encoding: str = Field(default="utf-8", description="File encoding")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        if ".." in v:
            raise ValueError("Path cannot contain '..'")
        return v


class FileWriteOutput(BaseModel):
    """Output schema for file write operation."""
    success: bool
    path: str
    bytes_written: int = 0
    error: Optional[str] = None


class FileWriteTool:
    """
    Write content to a file.

    Creates parent directories automatically.
    Overwrites existing files.
    """

    name: str = "file_write"
    description: str = "Write content to a file, creating directories if needed"
    input_schema = FileWriteInput
    output_schema = FileWriteOutput

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute(self, input_data: FileWriteInput) -> FileWriteOutput:
        """Execute file write operation."""
        try:
            full_path = self._resolve_path(input_data.path)

            # Create directories
            if input_data.create_dirs:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write file
            with open(full_path, "w", encoding=input_data.encoding) as f:
                f.write(input_data.content)

            return FileWriteOutput(
                success=True,
                path=input_data.path,
                bytes_written=len(input_data.content.encode(input_data.encoding)),
            )

        except Exception as e:
            return FileWriteOutput(
                success=False,
                path=input_data.path,
                error=str(e),
            )

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)


class FileEditInput(BaseModel):
    """Input schema for file edit operation."""
    path: str = Field(description="Path to file to edit")
    old_text: str = Field(description="Text to find and replace")
    new_text: str = Field(description="Replacement text")
    occurrence: int = Field(
        default=1,
        ge=0,
        description="Which occurrence to replace (0=all, 1=first, etc.)"
    )
    encoding: str = Field(default="utf-8", description="File encoding")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        if ".." in v:
            raise ValueError("Path cannot contain '..'")
        return v


class FileEditOutput(BaseModel):
    """Output schema for file edit operation."""
    success: bool
    path: str
    replacements: int = 0
    diff: Optional[str] = None
    error: Optional[str] = None


class FileEditTool:
    """
    Edit a file by replacing text.

    Supports replacing specific occurrences or all occurrences.
    Returns a diff of changes made.
    """

    name: str = "file_edit"
    description: str = "Edit a file by finding and replacing text"
    input_schema = FileEditInput
    output_schema = FileEditOutput

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute(self, input_data: FileEditInput) -> FileEditOutput:
        """Execute file edit operation."""
        try:
            full_path = self._resolve_path(input_data.path)

            if not os.path.exists(full_path):
                return FileEditOutput(
                    success=False,
                    path=input_data.path,
                    error=f"File not found: {input_data.path}",
                )

            # Read original content
            with open(full_path, "r", encoding=input_data.encoding) as f:
                original = f.read()

            # Check if old_text exists
            if input_data.old_text not in original:
                return FileEditOutput(
                    success=False,
                    path=input_data.path,
                    error="Text to replace not found in file",
                )

            # Perform replacement
            if input_data.occurrence == 0:
                # Replace all
                new_content = original.replace(input_data.old_text, input_data.new_text)
                replacements = original.count(input_data.old_text)
            else:
                # Replace specific occurrence
                parts = original.split(input_data.old_text)
                if len(parts) <= input_data.occurrence:
                    return FileEditOutput(
                        success=False,
                        path=input_data.path,
                        error=f"Occurrence {input_data.occurrence} not found",
                    )

                new_parts = []
                for i, part in enumerate(parts):
                    new_parts.append(part)
                    if i < len(parts) - 1:
                        if i + 1 == input_data.occurrence:
                            new_parts.append(input_data.new_text)
                        else:
                            new_parts.append(input_data.old_text)

                new_content = "".join(new_parts)
                replacements = 1

            # Generate diff
            diff = self._generate_diff(original, new_content, input_data.path)

            # Write new content
            with open(full_path, "w", encoding=input_data.encoding) as f:
                f.write(new_content)

            return FileEditOutput(
                success=True,
                path=input_data.path,
                replacements=replacements,
                diff=diff,
            )

        except Exception as e:
            return FileEditOutput(
                success=False,
                path=input_data.path,
                error=str(e),
            )

    def _generate_diff(self, old: str, new: str, path: str) -> str:
        """Generate unified diff."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )

        return "".join(diff)

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)


class FileSearchInput(BaseModel):
    """Input schema for file search operation."""
    pattern: str = Field(description="Search pattern (glob or regex)")
    path: str = Field(default=".", description="Directory to search in")
    content_pattern: Optional[str] = Field(
        default=None,
        description="Pattern to search within files"
    )
    max_results: int = Field(default=100, ge=1, le=1000)
    recursive: bool = Field(default=True)
    include_hidden: bool = Field(default=False)


class FileSearchOutput(BaseModel):
    """Output schema for file search operation."""
    success: bool
    matches: List[str] = []
    match_count: int = 0
    content_matches: List[Dict[str, Any]] = []
    error: Optional[str] = None


class FileSearchTool:
    """
    Search for files by name pattern.

    Supports glob patterns and optional content search.
    """

    name: str = "file_search"
    description: str = "Search for files matching a pattern, optionally searching content"
    input_schema = FileSearchInput
    output_schema = FileSearchOutput

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute(self, input_data: FileSearchInput) -> FileSearchOutput:
        """Execute file search operation."""
        try:
            search_path = self._resolve_path(input_data.path)

            if not os.path.exists(search_path):
                return FileSearchOutput(
                    success=False,
                    error=f"Path not found: {input_data.path}",
                )

            # Find matching files
            if input_data.recursive:
                pattern = os.path.join(search_path, "**", input_data.pattern)
                matches = glob_module.glob(pattern, recursive=True)
            else:
                pattern = os.path.join(search_path, input_data.pattern)
                matches = glob_module.glob(pattern)

            # Filter hidden files
            if not input_data.include_hidden:
                matches = [
                    m for m in matches
                    if not any(part.startswith(".") for part in Path(m).parts)
                ]

            # Limit results
            matches = matches[:input_data.max_results]

            # Make paths relative
            matches = [os.path.relpath(m, self.workspace_root) for m in matches]

            # Search content if requested
            content_matches = []
            if input_data.content_pattern:
                import re
                content_regex = re.compile(input_data.content_pattern)

                for match in matches:
                    full_path = self._resolve_path(match)
                    if os.path.isfile(full_path):
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                for i, line in enumerate(f, 1):
                                    if content_regex.search(line):
                                        content_matches.append({
                                            "file": match,
                                            "line": i,
                                            "content": line.strip(),
                                        })
                        except (UnicodeDecodeError, PermissionError):
                            continue

            return FileSearchOutput(
                success=True,
                matches=matches,
                match_count=len(matches),
                content_matches=content_matches[:input_data.max_results],
            )

        except Exception as e:
            return FileSearchOutput(
                success=False,
                error=str(e),
            )

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)


class DirectoryListInput(BaseModel):
    """Input schema for directory list operation."""
    path: str = Field(default=".", description="Directory to list")
    recursive: bool = Field(default=False)
    include_hidden: bool = Field(default=False)
    pattern: Optional[str] = Field(default=None, description="Filter pattern")


class DirectoryListOutput(BaseModel):
    """Output schema for directory list operation."""
    success: bool
    entries: List[Dict[str, Any]] = []
    total_entries: int = 0
    error: Optional[str] = None


class DirectoryListTool:
    """
    List directory contents.

    Returns files and directories with metadata.
    """

    name: str = "directory_list"
    description: str = "List contents of a directory"
    input_schema = DirectoryListInput
    output_schema = DirectoryListOutput

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute(self, input_data: DirectoryListInput) -> DirectoryListOutput:
        """Execute directory list operation."""
        try:
            full_path = self._resolve_path(input_data.path)

            if not os.path.exists(full_path):
                return DirectoryListOutput(
                    success=False,
                    error=f"Path not found: {input_data.path}",
                )

            if not os.path.isdir(full_path):
                return DirectoryListOutput(
                    success=False,
                    error=f"Not a directory: {input_data.path}",
                )

            entries = []

            if input_data.recursive:
                for root, dirs, files in os.walk(full_path):
                    # Filter hidden
                    if not input_data.include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                        files = [f for f in files if not f.startswith(".")]

                    for name in dirs + files:
                        full = os.path.join(root, name)
                        rel_path = os.path.relpath(full, self.workspace_root)

                        # Apply pattern filter
                        if input_data.pattern and not fnmatch.fnmatch(name, input_data.pattern):
                            continue

                        entries.append(self._get_entry_info(full, rel_path))
            else:
                for name in os.listdir(full_path):
                    # Filter hidden
                    if not input_data.include_hidden and name.startswith("."):
                        continue

                    # Apply pattern filter
                    if input_data.pattern and not fnmatch.fnmatch(name, input_data.pattern):
                        continue

                    full = os.path.join(full_path, name)
                    rel_path = os.path.relpath(full, self.workspace_root)
                    entries.append(self._get_entry_info(full, rel_path))

            return DirectoryListOutput(
                success=True,
                entries=entries,
                total_entries=len(entries),
            )

        except Exception as e:
            return DirectoryListOutput(
                success=False,
                error=str(e),
            )

    def _get_entry_info(self, full_path: str, rel_path: str) -> Dict[str, Any]:
        """Get entry information."""
        stat = os.stat(full_path)
        return {
            "path": rel_path,
            "name": os.path.basename(full_path),
            "type": "directory" if os.path.isdir(full_path) else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)
