"""
Recursive Language Model (RLM) Implementation (arxiv:2512.24601).

Context stored as external environment object that LLM manipulates
through a Python REPL. Enables handling inputs 2 orders of magnitude
beyond model context windows.

Key features:
- No information-destroying summarization
- Deterministic retrieval via code execution
- Transparent reasoning traces through code execution logs
- Recursive sub-model invocation for chunked processing
"""

import ast
import asyncio
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from cli_agent.core.state import RLMContext


@dataclass
class REPLResult:
    """Result of executing code in the REPL."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0
    variables_modified: list[str] = field(default_factory=list)


@dataclass
class RecursiveCallResult:
    """Result of a recursive sub-model call."""
    chunk_id: int
    input_size: int
    output: str
    tokens_used: int = 0


class RLMEngine:
    """
    Recursive Language Model Engine.

    Implements the RLM paradigm where:
    1. Full context is stored as a variable in external Python REPL
    2. LLM writes code to inspect, slice, and decompose context
    3. LLM recursively invokes itself (or sub-models) on relevant snippets
    4. Results aggregate through the call tree
    """

    def __init__(
        self,
        llm_client: Any = None,
        sub_model_client: Optional[Any] = None,
        max_recursion_depth: int = 5,
        chunk_size: int = 4000,
        repl_timeout: int = 30,
        model: Optional[str] = None,
    ):
        # Support both llm_client (object) and model (string) initialization
        self.model = model

        # Create simple client wrapper if model string is provided but no client
        if llm_client is None and model is not None:
            import os
            import litellm

            class SimpleLLMClient:
                def __init__(self, model_name):
                    self.model = model_name
                    # Get API credentials from environment
                    self.api_key = (
                        os.getenv("ANTHROPIC_AUTH_TOKEN")
                        or os.getenv("ANTHROPIC_API_KEY")
                        or os.getenv("OPENAI_API_KEY")
                    )
                    self.api_base = (
                        os.getenv("ANTHROPIC_BASE_URL")
                        or os.getenv("OPENAI_API_BASE")
                    )

                async def complete(self, prompt, **kwargs):
                    call_kwargs = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if self.api_key:
                        call_kwargs["api_key"] = self.api_key
                    if self.api_base:
                        call_kwargs["api_base"] = self.api_base
                    call_kwargs.update(kwargs)

                    response = await litellm.acompletion(**call_kwargs)
                    return response.choices[0].message.content

            llm_client = SimpleLLMClient(model)

        self.llm = llm_client
        self.sub_model = sub_model_client or llm_client
        self.max_recursion_depth = max_recursion_depth
        self.chunk_size = chunk_size
        self.repl_timeout = repl_timeout

        # REPL environment
        self._repl_globals: dict[str, Any] = {}
        self._execution_log: list[dict[str, Any]] = []

        # Context storage
        self.context = RLMContext()

        # Recursion tracking
        self._current_depth = 0

    def initialize_context(
        self,
        files: Optional[dict[str, str]] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Initialize the context environment."""
        self.context = RLMContext(
            files=files or {},
            history=history or [],
        )

        # Update totals
        total_chars = sum(len(c) for c in self.context.files.values())
        total_chars += sum(len(str(h)) for h in self.context.history)
        self.context.total_chars = total_chars
        self.context.total_tokens_estimate = total_chars // 4

        # Make context available in REPL
        self._repl_globals = {
            "ctx": self.context,
            "files": self.context.files,
            "history": self.context.history,
            "search": self.context.search_files,
            "get_chunk": self.context.get_file_chunk,
            "filter_history": self.context.filter_history,
            "errors": self.context.get_recent_errors,
            "summarize": self.context.summarize_file,
            # Utility imports
            "re": __import__("re"),
            "json": __import__("json"),
            "datetime": datetime,
        }

    def add_file(self, path: str, content: str) -> None:
        """Add or update a file in context."""
        self.context.files[path] = content
        self._update_totals()

    def add_history(self, role: str, content: str) -> None:
        """Add a message to history."""
        self.context.history.append({"role": role, "content": content})
        self._update_totals()

    def add_tool_output(self, tool_name: str, output: Any) -> None:
        """Add a tool output to context."""
        self.context.tool_outputs.append({
            "tool": tool_name,
            "output": output,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _update_totals(self) -> None:
        """Update total character and token counts."""
        total_chars = sum(len(c) for c in self.context.files.values())
        total_chars += sum(len(str(h)) for h in self.context.history)
        self.context.total_chars = total_chars
        self.context.total_tokens_estimate = total_chars // 4

    # =========================================================================
    # REPL Execution
    # =========================================================================

    def execute_code(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment.

        This is how the LLM manipulates the context programmatically.
        """
        start_time = datetime.utcnow()

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Track modified variables
        vars_before = set(self._repl_globals.keys())

        try:
            # Parse to check for syntax errors
            ast.parse(code)

            # Execute with timeout
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._repl_globals)

            vars_after = set(self._repl_globals.keys())
            modified = list(vars_after - vars_before)

            output = stdout_capture.getvalue()
            if not output and 'result' in self._repl_globals:
                output = str(self._repl_globals['result'])

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = REPLResult(
                success=True,
                output=output,
                execution_time_ms=execution_time,
                variables_modified=modified,
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result = REPLResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time_ms=execution_time,
            )

        # Log execution
        self._execution_log.append({
            "code": code,
            "result": result.output if result.success else result.error,
            "success": result.success,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return result

    async def execute_async(self, code: str) -> REPLResult:
        """Execute code asynchronously with timeout."""
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.execute_code, code
                ),
                timeout=self.repl_timeout,
            )
            return result
        except asyncio.TimeoutError:
            return REPLResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.repl_timeout} seconds",
            )

    # =========================================================================
    # Recursive Processing
    # =========================================================================

    async def process_with_recursion(
        self,
        query: str,
        context_key: str = "files",
        aggregation_strategy: str = "hierarchical",
    ) -> str:
        """
        Process a query using recursive sub-model calls.

        For large contexts, splits into chunks, processes each with
        a sub-model, and aggregates results.
        """
        self._current_depth = 0

        # Get the context to process
        if context_key == "files":
            context_items = list(self.context.files.items())
        elif context_key == "history":
            context_items = [(str(i), str(h)) for i, h in enumerate(self.context.history)]
        else:
            context_items = [(context_key, str(self._repl_globals.get(context_key, "")))]

        # Check if we need to split
        total_size = sum(len(str(v)) for _, v in context_items)

        if total_size < self.chunk_size * 2:
            # Small enough to process directly
            return await self._process_direct(query, context_items)

        # Need recursive processing
        return await self._process_recursive(
            query=query,
            items=context_items,
            depth=0,
            aggregation_strategy=aggregation_strategy,
        )

    async def _process_direct(
        self,
        query: str,
        items: list[tuple[str, str]],
    ) -> str:
        """Process items directly without recursion."""
        context_str = "\n\n".join(f"=== {k} ===\n{v}" for k, v in items)

        prompt = f"""Analyze this context to answer the query.

Query: {query}

Context:
{context_str}

Answer:
"""
        return await self.llm.complete(prompt)

    async def _process_recursive(
        self,
        query: str,
        items: list[tuple[str, str]],
        depth: int,
        aggregation_strategy: str,
    ) -> str:
        """
        Process items recursively with sub-model calls.

        Strategy:
        1. Split items into chunks
        2. Process each chunk with sub-model
        3. Aggregate results
        4. If still too large, recurse
        """
        if depth >= self.max_recursion_depth:
            # Max depth reached, truncate and process
            truncated = [(k, v[:self.chunk_size]) for k, v in items[:5]]
            return await self._process_direct(query, truncated)

        # Split into chunks
        chunks = self._create_chunks(items)

        # Process each chunk with sub-model
        chunk_results: list[RecursiveCallResult] = []

        for i, chunk in enumerate(chunks):
            chunk_context = "\n\n".join(f"=== {k} ===\n{v}" for k, v in chunk)

            prompt = f"""Extract information relevant to this query from the context.

Query: {query}

Context (chunk {i+1}/{len(chunks)}):
{chunk_context}

Relevant information (be concise):
"""
            result = await self.sub_model.complete(prompt)

            chunk_results.append(RecursiveCallResult(
                chunk_id=i,
                input_size=len(chunk_context),
                output=result,
            ))

        # Aggregate results
        if aggregation_strategy == "hierarchical":
            aggregated = await self._aggregate_hierarchical(query, chunk_results, depth)
        else:
            aggregated = await self._aggregate_simple(query, chunk_results)

        return aggregated

    def _create_chunks(
        self,
        items: list[tuple[str, str]],
    ) -> list[list[tuple[str, str]]]:
        """Split items into chunks of approximately chunk_size."""
        chunks = []
        current_chunk = []
        current_size = 0

        for key, value in items:
            item_size = len(value)

            if item_size > self.chunk_size:
                # Single item too large, split it
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0

                # Split large item
                for i in range(0, len(value), self.chunk_size):
                    chunk_value = value[i:i + self.chunk_size]
                    chunks.append([(f"{key}[{i}:{i+len(chunk_value)}]", chunk_value)])

            elif current_size + item_size > self.chunk_size:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [(key, value)]
                current_size = item_size

            else:
                current_chunk.append((key, value))
                current_size += item_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def _aggregate_hierarchical(
        self,
        query: str,
        results: list[RecursiveCallResult],
        depth: int,
    ) -> str:
        """Aggregate results hierarchically (may recurse if needed)."""
        # Combine all results
        combined = "\n\n".join(
            f"[From chunk {r.chunk_id}]: {r.output}"
            for r in results
        )

        # Check if combined is still too large
        if len(combined) > self.chunk_size * 2 and depth < self.max_recursion_depth - 1:
            # Recurse on the aggregated results
            items = [(f"partial_{r.chunk_id}", r.output) for r in results]
            return await self._process_recursive(
                query=query,
                items=items,
                depth=depth + 1,
                aggregation_strategy="hierarchical",
            )

        # Final aggregation
        prompt = f"""Synthesize these partial results into a final answer.

Query: {query}

Partial results from analyzing different parts of the context:
{combined}

Final synthesized answer:
"""
        return await self.llm.complete(prompt)

    async def _aggregate_simple(
        self,
        query: str,
        results: list[RecursiveCallResult],
    ) -> str:
        """Simple concatenation aggregation."""
        combined = "\n\n".join(r.output for r in results)

        prompt = f"""Combine these findings into a coherent answer.

Query: {query}

Findings:
{combined}

Combined answer:
"""
        return await self.llm.complete(prompt)

    # =========================================================================
    # LLM-Driven Context Manipulation
    # =========================================================================

    async def query_context(self, question: str) -> str:
        """
        Answer a question about the context using RLM approach.

        The LLM generates code to explore the context, executes it,
        and uses results to formulate an answer.
        """
        # First, let LLM generate exploration code
        code_prompt = f"""You have access to a Python REPL with the following context:
- `ctx`: RLMContext object with files, history, tool_outputs
- `files`: dict of file_path -> content
- `history`: list of conversation messages
- `search(pattern)`: search file contents by regex pattern
- `get_chunk(path, start_line, end_line)`: get specific lines from a file
- `filter_history(keyword)`: filter history by keyword
- `errors(n)`: get last n error messages

Question: {question}

Write Python code to explore the context and find the answer.
Store the final result in a variable called `result`.
Only output the Python code, no explanations.

```python
"""
        code_response = await self.llm.complete(code_prompt)

        # Clean up the code
        code = code_response.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # Execute the code
        exec_result = await self.execute_async(code)

        if not exec_result.success:
            # If code failed, fall back to direct processing
            return await self.process_with_recursion(question)

        # Get the result and formulate answer
        result = self._repl_globals.get('result', exec_result.output)

        answer_prompt = f"""Based on this exploration of the context:

Question: {question}

Code executed:
```python
{code}
```

Result:
{result}

Provide a clear, concise answer to the question:
"""
        return await self.llm.complete(answer_prompt)

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Get the execution log for transparency."""
        return self._execution_log.copy()

    def clear_execution_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()

    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL environment."""
        return self._repl_globals.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the REPL environment."""
        self._repl_globals[name] = value

    def stats(self) -> dict[str, Any]:
        """Get RLM engine statistics."""
        return {
            "total_files": len(self.context.files),
            "total_history": len(self.context.history),
            "total_chars": self.context.total_chars,
            "total_tokens_estimate": self.context.total_tokens_estimate,
            "executions": len(self._execution_log),
            "variables": list(self._repl_globals.keys()),
        }
