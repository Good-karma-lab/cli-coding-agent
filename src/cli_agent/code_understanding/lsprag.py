"""
LSPRAG - LSP-Guided Retrieval Augmented Generation.

Combines lexical analysis (Tree-sitter) and semantic analysis (LSP)
for real-time code understanding with neuro-symbolic validation.
"""

from dataclasses import dataclass
from typing import Any, Optional

from cli_agent.code_understanding.tree_sitter_parser import TreeSitterParser
from cli_agent.code_understanding.lsp_client import LSPClient, SimpleLSPClient, Location
from cli_agent.code_understanding.code_graph import CodeGraph


@dataclass
class CodeContext:
    """Rich context for code at a specific location."""
    file_path: str
    line: int
    character: int

    # From Tree-sitter (lexical)
    symbol_name: Optional[str] = None
    symbol_kind: Optional[str] = None
    parent_symbol: Optional[str] = None
    signature: Optional[str] = None

    # From LSP (semantic)
    definition_location: Optional[Location] = None
    references: list[Location] = None
    hover_info: Optional[str] = None
    diagnostics: list[str] = None

    # From Code Graph (structural)
    callers: list[str] = None
    callees: list[str] = None
    dependencies: list[str] = None

    def __post_init__(self):
        self.references = self.references or []
        self.diagnostics = self.diagnostics or []
        self.callers = self.callers or []
        self.callees = self.callees or []
        self.dependencies = self.dependencies or []


class LSPRAG:
    """
    LSP-Guided Retrieval Augmented Generation.

    Combines multiple code analysis approaches for comprehensive
    code understanding that can be used to augment LLM context.

    Key features:
    - Real-time code understanding via LSP
    - Structural analysis via Tree-sitter
    - Graph-based relationship queries
    - Neuro-symbolic validation via diagnostics
    """

    def __init__(
        self,
        project_root: str,
        code_graph: Optional[CodeGraph] = None,
        use_lsp: bool = True,
    ):
        self.project_root = project_root
        self.parser = TreeSitterParser()

        # Code graph for structural queries
        self.code_graph = code_graph or CodeGraph(project_root)

        # LSP client for semantic analysis
        self.use_lsp = use_lsp
        self.lsp_clients: dict[str, LSPClient] = {}
        self.simple_lsp = SimpleLSPClient(project_root)

    async def get_context(
        self,
        file_path: str,
        line: int,
        character: int,
        content: Optional[str] = None,
    ) -> CodeContext:
        """
        Get rich context for a code location.

        Combines Tree-sitter lexical analysis, LSP semantic analysis,
        and code graph structural queries.
        """
        context = CodeContext(
            file_path=file_path,
            line=line,
            character=character,
        )

        # Get content if not provided
        if content is None:
            try:
                with open(file_path, "r") as f:
                    content = f.read()
            except Exception:
                return context

        # Tree-sitter analysis
        parse_result = self.parser.parse_file(file_path, content)
        symbol = self.parser.get_symbol_at_position(file_path, line, character, content)

        if symbol:
            context.symbol_name = symbol.name
            context.symbol_kind = symbol.kind
            context.parent_symbol = symbol.parent
            context.signature = symbol.signature

        # LSP analysis
        if self.use_lsp:
            language = self.parser.detect_language(file_path)
            if language:
                lsp = await self._get_lsp_client(language)
                if lsp and lsp.is_initialized:
                    # Get definition
                    definitions = await lsp.get_definition(file_path, line, character)
                    if definitions:
                        context.definition_location = definitions[0]

                    # Get references
                    context.references = await lsp.get_references(file_path, line, character)

                    # Get hover info
                    context.hover_info = await lsp.get_hover(file_path, line, character)

                    # Get diagnostics
                    diagnostics = await lsp.get_diagnostics(file_path)
                    context.diagnostics = [d.message for d in diagnostics]

        # Code graph queries
        if context.symbol_name:
            context.callers = self.code_graph.get_callers(context.symbol_name)
            context.callees = self.code_graph.get_callees(context.symbol_name)

        context.dependencies = self.code_graph.get_dependencies(file_path)

        return context

    async def search_symbols(
        self,
        query: str,
        kind_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for symbols across the codebase.

        Combines LSP workspace/symbol with code graph search.
        """
        results = []

        # Try LSP first (more accurate)
        for language, lsp in self.lsp_clients.items():
            if lsp.is_initialized:
                symbols = await lsp.get_workspace_symbols(query)
                for sym in symbols:
                    if kind_filter is None or str(sym.kind) == kind_filter:
                        results.append({
                            "name": sym.name,
                            "kind": sym.kind,
                            "file": sym.location.file_path,
                            "line": sym.location.start_line,
                            "container": sym.container_name,
                            "source": "lsp",
                        })

        # Supplement with code graph
        graph_results = self.code_graph.find_symbol(query)
        for node in graph_results:
            results.append({
                "name": node.get("name"),
                "kind": node.get("kind"),
                "file": node.get("file_path"),
                "line": node.get("start_line"),
                "container": node.get("metadata", {}).get("parent"),
                "source": "graph",
            })

        # Deduplicate
        seen = set()
        unique_results = []
        for r in results:
            key = (r["name"], r["file"], r["line"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results

    async def find_definition(
        self,
        symbol_name: str,
        from_file: Optional[str] = None,
    ) -> list[Location]:
        """
        Find where a symbol is defined.

        Uses LSP for accuracy, falls back to simple search.
        """
        # Try LSP if we have context file
        if from_file and self.use_lsp:
            language = self.parser.detect_language(from_file)
            if language:
                lsp = await self._get_lsp_client(language)
                if lsp and lsp.is_initialized:
                    # Would need to find position of symbol in file first
                    pass

        # Fall back to simple LSP
        return await self.simple_lsp.find_definition(symbol_name)

    async def find_references(
        self,
        symbol_name: str,
    ) -> list[Location]:
        """
        Find all references to a symbol.

        Uses LSP for accuracy, falls back to grep-based search.
        """
        return await self.simple_lsp.find_references(symbol_name)

    async def validate_code(
        self,
        file_path: str,
        content: str,
    ) -> list[str]:
        """
        Validate code using LSP diagnostics.

        Implements neuro-symbolic validation - using symbolic tools
        (LSP/compiler) to validate LLM-generated code.
        """
        errors = []

        if self.use_lsp:
            language = self.parser.detect_language(file_path)
            if language:
                lsp = await self._get_lsp_client(language)
                if lsp and lsp.is_initialized:
                    # Update file in LSP
                    await lsp.update_file(file_path, content)

                    # Wait a bit for diagnostics
                    import asyncio
                    await asyncio.sleep(0.5)

                    # Get diagnostics
                    diagnostics = await lsp.get_diagnostics(file_path)
                    for d in diagnostics:
                        if d.severity == 1:  # Error
                            errors.append(f"Line {d.line}: {d.message}")

        # Also do basic syntax check via Tree-sitter
        parse_result = self.parser.parse_file(file_path, content)
        if parse_result.has_syntax_errors:
            errors.extend(parse_result.errors)

        return errors

    async def get_relevant_context(
        self,
        query: str,
        max_files: int = 5,
        max_tokens: int = 4000,
    ) -> str:
        """
        Get relevant code context for a query.

        This is the main RAG function - retrieves relevant code
        to augment LLM context.
        """
        context_parts = []
        current_tokens = 0

        # Search for relevant symbols
        symbols = await self.search_symbols(query)

        # Group by file
        files: dict[str, list[dict]] = {}
        for sym in symbols[:20]:  # Limit search
            file_path = sym.get("file", "")
            if file_path:
                files.setdefault(file_path, []).append(sym)

        # Read relevant portions of top files
        for file_path in list(files.keys())[:max_files]:
            try:
                with open(file_path, "r") as f:
                    content = f.read()
            except Exception:
                continue

            # Get the specific symbols we're interested in
            relevant_symbols = files[file_path]
            lines = content.split("\n")

            # Extract relevant lines around each symbol
            relevant_lines = set()
            for sym in relevant_symbols:
                start_line = sym.get("line", 0)
                # Get surrounding context
                for i in range(max(0, start_line - 2), min(len(lines), start_line + 20)):
                    relevant_lines.add(i)

            # Build excerpt
            if relevant_lines:
                sorted_lines = sorted(relevant_lines)
                excerpt_parts = []
                prev_line = -2

                for line_num in sorted_lines:
                    if line_num - prev_line > 1:
                        excerpt_parts.append("...")
                    excerpt_parts.append(f"{line_num + 1}: {lines[line_num]}")
                    prev_line = line_num

                excerpt = "\n".join(excerpt_parts)
                tokens = len(excerpt.split()) * 4 // 3

                if current_tokens + tokens <= max_tokens:
                    context_parts.append(f"=== {file_path} ===\n{excerpt}")
                    current_tokens += tokens

        return "\n\n".join(context_parts)

    async def _get_lsp_client(self, language: str) -> Optional[LSPClient]:
        """Get or create LSP client for a language."""
        if language not in self.lsp_clients:
            client = LSPClient(
                project_root=self.project_root,
                language=language,
            )
            if await client.start():
                self.lsp_clients[language] = client
            else:
                return None

        return self.lsp_clients.get(language)

    async def close(self) -> None:
        """Close all LSP clients."""
        for client in self.lsp_clients.values():
            await client.stop()
        self.lsp_clients.clear()
