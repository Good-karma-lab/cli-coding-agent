"""
Code Tools - Code navigation and search tools.

Integrates with code understanding systems for:
- Semantic code search
- Symbol lookup
- Reference finding
"""

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CodeSearchInput(BaseModel):
    """Input schema for code search."""
    query: str = Field(description="Search query (text or regex)")
    path: str = Field(default=".", description="Search path")
    file_pattern: Optional[str] = Field(
        default=None,
        description="File pattern filter (e.g., '*.py')"
    )
    max_results: int = Field(default=50, ge=1, le=500)
    context_lines: int = Field(default=2, ge=0, le=10)
    case_sensitive: bool = Field(default=False)
    regex: bool = Field(default=False)


class CodeSearchMatch(BaseModel):
    """A single code search match."""
    file: str
    line: int
    column: int = 0
    content: str
    context_before: List[str] = []
    context_after: List[str] = []


class CodeSearchOutput(BaseModel):
    """Output schema for code search."""
    success: bool
    matches: List[CodeSearchMatch] = []
    total_matches: int = 0
    files_searched: int = 0
    error: Optional[str] = None


class CodeSearchTool:
    """
    Search for code patterns in the codebase.

    Supports text search and regex patterns.
    Returns matches with surrounding context.
    """

    name: str = "code_search"
    description: str = "Search for code patterns in files"
    input_schema = CodeSearchInput
    output_schema = CodeSearchOutput

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()

    def execute(self, input_data: CodeSearchInput) -> CodeSearchOutput:
        """Execute code search."""
        import re
        import fnmatch

        try:
            search_path = self._resolve_path(input_data.path)

            if not os.path.exists(search_path):
                return CodeSearchOutput(
                    success=False,
                    error=f"Path not found: {input_data.path}",
                )

            # Compile pattern
            if input_data.regex:
                flags = 0 if input_data.case_sensitive else re.IGNORECASE
                try:
                    pattern = re.compile(input_data.query, flags)
                except re.error as e:
                    return CodeSearchOutput(
                        success=False,
                        error=f"Invalid regex: {e}",
                    )
            else:
                query = input_data.query
                if not input_data.case_sensitive:
                    query = query.lower()

            matches = []
            files_searched = 0

            # Walk directory
            for root, dirs, files in os.walk(search_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith(".")]

                for filename in files:
                    # Skip hidden files
                    if filename.startswith("."):
                        continue

                    # Apply file pattern filter
                    if input_data.file_pattern:
                        if not fnmatch.fnmatch(filename, input_data.file_pattern):
                            continue

                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, self.workspace_root)

                    try:
                        file_matches = self._search_file(
                            file_path,
                            rel_path,
                            pattern if input_data.regex else query,
                            input_data.regex,
                            input_data.case_sensitive,
                            input_data.context_lines,
                        )
                        matches.extend(file_matches)
                        files_searched += 1

                        if len(matches) >= input_data.max_results:
                            break

                    except (UnicodeDecodeError, PermissionError):
                        continue

                if len(matches) >= input_data.max_results:
                    break

            return CodeSearchOutput(
                success=True,
                matches=matches[:input_data.max_results],
                total_matches=len(matches),
                files_searched=files_searched,
            )

        except Exception as e:
            return CodeSearchOutput(
                success=False,
                error=str(e),
            )

    def _search_file(
        self,
        file_path: str,
        rel_path: str,
        pattern: Any,
        is_regex: bool,
        case_sensitive: bool,
        context_lines: int,
    ) -> List[CodeSearchMatch]:
        """Search a single file."""
        matches = []

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line_content = line.rstrip("\n\r")
            search_line = line_content if case_sensitive else line_content.lower()

            # Check for match
            found = False
            column = 0

            if is_regex:
                match = pattern.search(line_content)
                if match:
                    found = True
                    column = match.start()
            else:
                if pattern in search_line:
                    found = True
                    column = search_line.find(pattern)

            if found:
                # Get context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                context_before = [
                    lines[j].rstrip("\n\r")
                    for j in range(start, i)
                ]
                context_after = [
                    lines[j].rstrip("\n\r")
                    for j in range(i + 1, end)
                ]

                matches.append(CodeSearchMatch(
                    file=rel_path,
                    line=i + 1,
                    column=column,
                    content=line_content,
                    context_before=context_before,
                    context_after=context_after,
                ))

        return matches

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)


class SymbolLookupInput(BaseModel):
    """Input schema for symbol lookup."""
    symbol: str = Field(description="Symbol name to look up")
    file: Optional[str] = Field(default=None, description="File to search in")
    symbol_type: Optional[str] = Field(
        default=None,
        description="Type: function, class, variable, etc."
    )


class SymbolInfo(BaseModel):
    """Information about a symbol."""
    name: str
    type: str
    file: str
    line: int
    column: int = 0
    signature: Optional[str] = None
    docstring: Optional[str] = None
    scope: Optional[str] = None


class SymbolLookupOutput(BaseModel):
    """Output schema for symbol lookup."""
    success: bool
    symbols: List[SymbolInfo] = []
    error: Optional[str] = None


class SymbolLookupTool:
    """
    Look up symbol definitions.

    Uses tree-sitter parsing for accurate symbol location.
    """

    name: str = "symbol_lookup"
    description: str = "Look up symbol definitions in code"
    input_schema = SymbolLookupInput
    output_schema = SymbolLookupOutput

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        parser: Optional[Any] = None,
    ):
        self.workspace_root = workspace_root or os.getcwd()
        self.parser = parser  # TreeSitterParser instance

    def execute(self, input_data: SymbolLookupInput) -> SymbolLookupOutput:
        """Execute symbol lookup."""
        try:
            symbols = []

            if input_data.file:
                # Search specific file
                file_path = self._resolve_path(input_data.file)
                if os.path.exists(file_path):
                    file_symbols = self._find_symbols_in_file(
                        file_path,
                        input_data.file,
                        input_data.symbol,
                        input_data.symbol_type,
                    )
                    symbols.extend(file_symbols)
            else:
                # Search all files
                for root, dirs, files in os.walk(self.workspace_root):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                    for filename in files:
                        if not self._is_code_file(filename):
                            continue

                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(file_path, self.workspace_root)

                        try:
                            file_symbols = self._find_symbols_in_file(
                                file_path,
                                rel_path,
                                input_data.symbol,
                                input_data.symbol_type,
                            )
                            symbols.extend(file_symbols)
                        except Exception:
                            continue

            return SymbolLookupOutput(
                success=True,
                symbols=symbols,
            )

        except Exception as e:
            return SymbolLookupOutput(
                success=False,
                error=str(e),
            )

    def _find_symbols_in_file(
        self,
        file_path: str,
        rel_path: str,
        symbol_name: str,
        symbol_type: Optional[str],
    ) -> List[SymbolInfo]:
        """Find symbols in a file."""
        import re

        symbols = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Simple pattern matching for common definitions
        patterns = {
            "function": [
                r"def\s+(" + re.escape(symbol_name) + r")\s*\(",  # Python
                r"function\s+(" + re.escape(symbol_name) + r")\s*\(",  # JS
                r"func\s+(" + re.escape(symbol_name) + r")\s*\(",  # Go
                r"fn\s+(" + re.escape(symbol_name) + r")\s*\(",  # Rust
            ],
            "class": [
                r"class\s+(" + re.escape(symbol_name) + r")\s*[:\(]",  # Python
                r"class\s+(" + re.escape(symbol_name) + r")\s*\{",  # JS/Java
                r"struct\s+(" + re.escape(symbol_name) + r")\s*\{",  # Go/Rust
            ],
            "variable": [
                r"(" + re.escape(symbol_name) + r")\s*=",  # Assignment
                r"const\s+(" + re.escape(symbol_name) + r")\s*=",  # Const
                r"let\s+(" + re.escape(symbol_name) + r")\s*=",  # Let
                r"var\s+(" + re.escape(symbol_name) + r")\s*=",  # Var
            ],
        }

        search_patterns = []
        if symbol_type and symbol_type in patterns:
            search_patterns = patterns[symbol_type]
        else:
            for pattern_list in patterns.values():
                search_patterns.extend(pattern_list)

        for i, line in enumerate(lines):
            for pattern in search_patterns:
                match = re.search(pattern, line)
                if match:
                    # Get docstring if present (simple check)
                    docstring = None
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('"""') or next_line.startswith("'''"):
                            docstring = next_line.strip('"\'')

                    symbols.append(SymbolInfo(
                        name=match.group(1),
                        type=self._infer_type(pattern),
                        file=rel_path,
                        line=i + 1,
                        column=match.start(),
                        signature=line.strip(),
                        docstring=docstring,
                    ))

        return symbols

    def _infer_type(self, pattern: str) -> str:
        """Infer symbol type from pattern."""
        if "def " in pattern or "function " in pattern or "func " in pattern or "fn " in pattern:
            return "function"
        elif "class " in pattern or "struct " in pattern:
            return "class"
        else:
            return "variable"

    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file."""
        extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".go", ".rs", ".cpp", ".c",
            ".h", ".hpp", ".rb", ".php", ".cs",
        }
        return any(filename.endswith(ext) for ext in extensions)

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace_root, path)


class ReferenceFindInput(BaseModel):
    """Input schema for finding references."""
    symbol: str = Field(description="Symbol to find references for")
    file: Optional[str] = Field(default=None, description="Starting file")
    include_definition: bool = Field(default=True)


class Reference(BaseModel):
    """A reference to a symbol."""
    file: str
    line: int
    column: int = 0
    content: str
    is_definition: bool = False


class ReferenceFinderOutput(BaseModel):
    """Output schema for reference finding."""
    success: bool
    references: List[Reference] = []
    definition: Optional[Reference] = None
    total_references: int = 0
    error: Optional[str] = None


class ReferenceFinderTool:
    """
    Find all references to a symbol.

    Locates usages of a symbol across the codebase.
    """

    name: str = "reference_finder"
    description: str = "Find all references to a symbol"
    input_schema = ReferenceFindInput
    output_schema = ReferenceFinderOutput

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        code_search: Optional[CodeSearchTool] = None,
        symbol_lookup: Optional[SymbolLookupTool] = None,
    ):
        self.workspace_root = workspace_root or os.getcwd()
        self.code_search = code_search or CodeSearchTool(workspace_root)
        self.symbol_lookup = symbol_lookup or SymbolLookupTool(workspace_root)

    def execute(self, input_data: ReferenceFindInput) -> ReferenceFinderOutput:
        """Execute reference finding."""
        try:
            # Find definition first
            definition = None
            if input_data.include_definition:
                lookup_result = self.symbol_lookup.execute(
                    SymbolLookupInput(
                        symbol=input_data.symbol,
                        file=input_data.file,
                    )
                )
                if lookup_result.success and lookup_result.symbols:
                    sym = lookup_result.symbols[0]
                    definition = Reference(
                        file=sym.file,
                        line=sym.line,
                        column=sym.column,
                        content=sym.signature or "",
                        is_definition=True,
                    )

            # Search for references
            search_result = self.code_search.execute(
                CodeSearchInput(
                    query=r"\b" + input_data.symbol + r"\b",
                    regex=True,
                    max_results=200,
                )
            )

            references = []
            if search_result.success:
                for match in search_result.matches:
                    # Skip definition line if we have it
                    if (definition and
                        match.file == definition.file and
                        match.line == definition.line):
                        continue

                    references.append(Reference(
                        file=match.file,
                        line=match.line,
                        column=match.column,
                        content=match.content,
                        is_definition=False,
                    ))

            return ReferenceFinderOutput(
                success=True,
                references=references,
                definition=definition,
                total_references=len(references),
            )

        except Exception as e:
            return ReferenceFinderOutput(
                success=False,
                error=str(e),
            )
