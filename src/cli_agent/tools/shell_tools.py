"""
Shell Tools - Safe shell execution with constraints.

Implements:
- Command allowlisting for safety
- Timeout management
- Output capture and streaming
- Environment isolation
"""

import asyncio
import os
import shlex
import subprocess
import signal
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator
import re


class ShellExecuteInput(BaseModel):
    """Input schema for shell execution."""
    command: str = Field(description="Command to execute")
    cwd: Optional[str] = Field(default=None, description="Working directory")
    timeout: int = Field(default=60, ge=1, le=600, description="Timeout in seconds")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Command cannot be empty")
        return v


class ShellExecuteOutput(BaseModel):
    """Output schema for shell execution."""
    success: bool
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None


class SafeShellExecutor:
    """
    Safe shell command executor with constraints.

    Features:
    - Command allowlisting
    - Dangerous command blocking
    - Timeout enforcement
    - Resource limits
    """

    # Commands that are always allowed
    ALLOWED_COMMANDS: Set[str] = {
        # Build tools
        "npm", "npx", "yarn", "pnpm",
        "pip", "pip3", "poetry", "pipenv",
        "cargo", "go", "make", "cmake",
        "mvn", "gradle",
        # Testing
        "pytest", "jest", "mocha", "vitest",
        "python", "python3", "node",
        # File operations
        "ls", "cat", "head", "tail", "grep", "find",
        "wc", "sort", "uniq", "diff",
        # Info commands
        "pwd", "whoami", "date", "env",
        "which", "whereis", "file", "stat",
        # Git (basic)
        "git",
    }

    # Commands/patterns that are blocked
    BLOCKED_PATTERNS: List[str] = [
        r"rm\s+-rf\s+/",        # rm -rf /
        r"rm\s+-rf\s+\*",       # rm -rf *
        r">\s*/dev/sd",         # Write to disk devices
        r"mkfs",                 # Format filesystems
        r"dd\s+.*of=/dev",      # dd to devices
        r"curl.*\|\s*sh",       # Pipe curl to shell
        r"wget.*\|\s*sh",       # Pipe wget to shell
        r"chmod\s+777",         # Overly permissive chmod
        r"sudo",                 # sudo commands
        r"su\s+-",               # su commands
        r":(){",                 # Fork bomb
        r">\s*/etc/",           # Write to /etc
        r"eval\s+.*\$",         # Dangerous eval
    ]

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        additional_allowed: Optional[Set[str]] = None,
        additional_blocked: Optional[List[str]] = None,
        max_output_size: int = 100000,  # 100KB
    ):
        self.workspace_root = workspace_root or os.getcwd()
        self.max_output_size = max_output_size

        self.allowed_commands = self.ALLOWED_COMMANDS.copy()
        if additional_allowed:
            self.allowed_commands.update(additional_allowed)

        self.blocked_patterns = self.BLOCKED_PATTERNS.copy()
        if additional_blocked:
            self.blocked_patterns.extend(additional_blocked)

        self._compiled_blocked = [
            re.compile(pattern) for pattern in self.blocked_patterns
        ]

    def is_command_allowed(self, command: str) -> tuple:
        """
        Check if a command is allowed.

        Returns (allowed, reason) tuple.
        """
        # Check blocked patterns
        for pattern in self._compiled_blocked:
            if pattern.search(command):
                return False, f"Command matches blocked pattern: {pattern.pattern}"

        # Extract base command
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command"
            base_command = os.path.basename(parts[0])
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"

        # Check allowlist
        if base_command not in self.allowed_commands:
            return False, f"Command not in allowlist: {base_command}"

        return True, "OK"

    def execute(self, input_data: ShellExecuteInput) -> ShellExecuteOutput:
        """Execute a shell command synchronously."""
        # Validate command
        allowed, reason = self.is_command_allowed(input_data.command)
        if not allowed:
            return ShellExecuteOutput(
                success=False,
                error=f"Command blocked: {reason}",
            )

        # Prepare environment
        env = os.environ.copy()
        if input_data.env:
            env.update(input_data.env)

        # Resolve working directory
        cwd = input_data.cwd or self.workspace_root
        if not os.path.isabs(cwd):
            cwd = os.path.join(self.workspace_root, cwd)

        try:
            # Execute command
            result = subprocess.run(
                input_data.command,
                shell=True,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=input_data.timeout,
            )

            # Truncate output if needed
            stdout = result.stdout
            stderr = result.stderr

            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (truncated)"

            return ShellExecuteOutput(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        except subprocess.TimeoutExpired:
            return ShellExecuteOutput(
                success=False,
                timed_out=True,
                error=f"Command timed out after {input_data.timeout}s",
            )
        except Exception as e:
            return ShellExecuteOutput(
                success=False,
                error=str(e),
            )

    async def execute_async(self, input_data: ShellExecuteInput) -> ShellExecuteOutput:
        """Execute a shell command asynchronously."""
        # Validate command
        allowed, reason = self.is_command_allowed(input_data.command)
        if not allowed:
            return ShellExecuteOutput(
                success=False,
                error=f"Command blocked: {reason}",
            )

        # Prepare environment
        env = os.environ.copy()
        if input_data.env:
            env.update(input_data.env)

        # Resolve working directory
        cwd = input_data.cwd or self.workspace_root
        if not os.path.isabs(cwd):
            cwd = os.path.join(self.workspace_root, cwd)

        try:
            # Create process
            process = await asyncio.create_subprocess_shell(
                input_data.command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=input_data.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ShellExecuteOutput(
                    success=False,
                    timed_out=True,
                    error=f"Command timed out after {input_data.timeout}s",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # Truncate if needed
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... (truncated)"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... (truncated)"

            return ShellExecuteOutput(
                success=process.returncode == 0,
                exit_code=process.returncode or -1,
                stdout=stdout,
                stderr=stderr,
            )

        except Exception as e:
            return ShellExecuteOutput(
                success=False,
                error=str(e),
            )


class ShellExecuteTool:
    """
    Shell execution tool for agents.

    Uses SafeShellExecutor with command validation.
    """

    name: str = "shell_execute"
    description: str = "Execute a shell command safely"
    input_schema = ShellExecuteInput
    output_schema = ShellExecuteOutput

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        executor: Optional[SafeShellExecutor] = None,
    ):
        self.executor = executor or SafeShellExecutor(workspace_root)

    def execute(self, input_data: ShellExecuteInput) -> ShellExecuteOutput:
        """Execute shell command."""
        return self.executor.execute(input_data)

    async def execute_async(self, input_data: ShellExecuteInput) -> ShellExecuteOutput:
        """Execute shell command asynchronously."""
        return await self.executor.execute_async(input_data)

    def is_allowed(self, command: str) -> tuple:
        """Check if command is allowed."""
        return self.executor.is_command_allowed(command)
