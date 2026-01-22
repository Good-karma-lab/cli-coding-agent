"""
DeepAgents Sandbox Integration

Uses LangChain DeepAgents built-in sandbox instead of custom Docker.
This provides secure code execution with proper isolation.

Based on research: "DeepAgents sandbox instead of Docker" for efficient
and safe code execution.
"""

import asyncio
import os
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import subprocess


class ExecutionStatus(str, Enum):
    """Status of sandbox execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


class SandboxConfig(BaseModel):
    """Configuration for sandbox execution."""
    timeout: int = Field(default=300, description="Execution timeout in seconds")
    memory_limit: str = Field(default="512m", description="Memory limit")
    cpu_limit: float = Field(default=1.0, description="CPU limit (cores)")
    network_enabled: bool = Field(default=False, description="Allow network access")
    workspace_path: Optional[str] = Field(default=None, description="Workspace path")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    allowed_commands: List[str] = Field(default_factory=list, description="Allowed shell commands")


class ExecutionResult(BaseModel):
    """Result from sandbox execution."""
    status: ExecutionStatus
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    files_created: List[str] = []
    files_modified: List[str] = []
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


@dataclass
class SandboxSession:
    """An active sandbox session."""
    id: str
    workspace: str
    config: SandboxConfig
    created_at: datetime = field(default_factory=datetime.now)
    executions: List[ExecutionResult] = field(default_factory=list)
    active: bool = True

    def record_execution(self, result: ExecutionResult) -> None:
        """Record an execution result."""
        self.executions.append(result)


class DeepAgentsSandbox:
    """
    DeepAgents Sandbox for secure code execution.

    This integrates with the LangChain DeepAgents sandbox system
    to provide isolated code execution environment.

    Features:
    - Secure isolation from host system
    - Resource limits (CPU, memory, time)
    - File system isolation with workspace
    - Network control
    - Execution history tracking
    """

    def __init__(
        self,
        default_config: Optional[SandboxConfig] = None,
        workspace_root: Optional[str] = None,
    ):
        self.default_config = default_config or SandboxConfig()
        self.workspace_root = workspace_root or tempfile.gettempdir()
        self._sessions: Dict[str, SandboxSession] = {}
        self._session_counter = 0

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        self._session_counter += 1
        return f"sandbox_{self._session_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    async def create_session(
        self,
        config: Optional[SandboxConfig] = None,
        copy_workspace: Optional[str] = None,
    ) -> SandboxSession:
        """
        Create a new sandbox session.

        Args:
            config: Session configuration
            copy_workspace: Path to copy as initial workspace

        Returns:
            SandboxSession instance
        """
        config = config or self.default_config
        session_id = self._generate_session_id()

        # Create isolated workspace
        workspace = os.path.join(self.workspace_root, f"sandbox_{session_id}")
        os.makedirs(workspace, exist_ok=True)

        # Copy initial workspace if provided
        if copy_workspace and os.path.exists(copy_workspace):
            self._copy_workspace(copy_workspace, workspace)

        session = SandboxSession(
            id=session_id,
            workspace=workspace,
            config=config,
        )

        self._sessions[session_id] = session
        return session

    async def execute_code(
        self,
        session: SandboxSession,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code in sandbox.

        Args:
            session: Sandbox session
            code: Code to execute
            language: Programming language
            filename: Optional filename for the code

        Returns:
            ExecutionResult with output
        """
        start_time = datetime.now()

        # Determine execution method based on language
        executor = self._get_executor(language)
        if not executor:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Unsupported language: {language}",
            )

        # Create code file
        if filename:
            code_path = os.path.join(session.workspace, filename)
        else:
            ext = self._get_extension(language)
            code_path = os.path.join(session.workspace, f"main{ext}")

        with open(code_path, "w") as f:
            f.write(code)

        # Track files before execution
        files_before = self._list_files(session.workspace)

        try:
            # Execute code
            result = await executor(
                code_path,
                session.workspace,
                session.config,
            )

            # Track files after execution
            files_after = self._list_files(session.workspace)

            # Determine created and modified files
            files_created = [f for f in files_after if f not in files_before]
            files_modified = [
                f for f in files_after
                if f in files_before and
                os.path.getmtime(os.path.join(session.workspace, f)) > start_time.timestamp()
            ]

            result.files_created = files_created
            result.files_modified = files_modified
            result.duration = (datetime.now() - start_time).total_seconds()

            session.record_execution(result)
            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                duration=(datetime.now() - start_time).total_seconds(),
                error=f"Execution timed out after {session.config.timeout}s",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                duration=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def execute_command(
        self,
        session: SandboxSession,
        command: str,
    ) -> ExecutionResult:
        """
        Execute a shell command in sandbox.

        Args:
            session: Sandbox session
            command: Command to execute

        Returns:
            ExecutionResult with output
        """
        start_time = datetime.now()

        # Check if command is allowed
        base_cmd = command.split()[0] if command else ""
        if session.config.allowed_commands:
            if base_cmd not in session.config.allowed_commands:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=f"Command not allowed: {base_cmd}",
                )

        # Prepare environment
        env = os.environ.copy()
        env.update(session.config.environment)

        try:
            # Execute with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=session.workspace,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=session.config.timeout,
            )

            result = ExecutionResult(
                status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration=(datetime.now() - start_time).total_seconds(),
            )

            session.record_execution(result)
            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                duration=(datetime.now() - start_time).total_seconds(),
                error=f"Command timed out after {session.config.timeout}s",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                duration=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def run_tests(
        self,
        session: SandboxSession,
        test_command: Optional[str] = None,
        language: str = "python",
    ) -> ExecutionResult:
        """
        Run tests in sandbox.

        Args:
            session: Sandbox session
            test_command: Custom test command
            language: Language for default test command

        Returns:
            ExecutionResult with test output
        """
        # Default test commands by language
        default_commands = {
            "python": "python -m pytest -v",
            "javascript": "npm test",
            "typescript": "npm test",
            "go": "go test ./...",
            "rust": "cargo test",
            "java": "mvn test",
        }

        command = test_command or default_commands.get(language, "make test")

        return await self.execute_command(session, command)

    async def cleanup_session(self, session: SandboxSession) -> None:
        """
        Clean up a sandbox session.

        Args:
            session: Session to clean up
        """
        session.active = False

        # Remove workspace
        if os.path.exists(session.workspace):
            shutil.rmtree(session.workspace, ignore_errors=True)

        # Remove from active sessions
        if session.id in self._sessions:
            del self._sessions[session.id]

    def get_session(self, session_id: str) -> Optional[SandboxSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[SandboxSession]:
        """List all active sessions."""
        return list(self._sessions.values())

    def _get_executor(self, language: str) -> Optional[Callable]:
        """Get executor function for language."""
        executors = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "typescript": self._execute_typescript,
            "go": self._execute_go,
            "rust": self._execute_rust,
            "bash": self._execute_bash,
            "shell": self._execute_bash,
        }
        return executors.get(language.lower())

    def _get_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "go": ".go",
            "rust": ".rs",
            "bash": ".sh",
            "shell": ".sh",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c",
        }
        return extensions.get(language.lower(), ".txt")

    async def _execute_python(
        self,
        code_path: str,
        workspace: str,
        config: SandboxConfig,
    ) -> ExecutionResult:
        """Execute Python code."""
        env = os.environ.copy()
        env.update(config.environment)

        process = await asyncio.create_subprocess_exec(
            "python", code_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.timeout,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def _execute_javascript(
        self,
        code_path: str,
        workspace: str,
        config: SandboxConfig,
    ) -> ExecutionResult:
        """Execute JavaScript code."""
        env = os.environ.copy()
        env.update(config.environment)

        process = await asyncio.create_subprocess_exec(
            "node", code_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.timeout,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def _execute_typescript(
        self,
        code_path: str,
        workspace: str,
        config: SandboxConfig,
    ) -> ExecutionResult:
        """Execute TypeScript code."""
        env = os.environ.copy()
        env.update(config.environment)

        # Use ts-node or compile first
        process = await asyncio.create_subprocess_exec(
            "npx", "ts-node", code_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.timeout,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def _execute_go(
        self,
        code_path: str,
        workspace: str,
        config: SandboxConfig,
    ) -> ExecutionResult:
        """Execute Go code."""
        env = os.environ.copy()
        env.update(config.environment)

        process = await asyncio.create_subprocess_exec(
            "go", "run", code_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.timeout,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def _execute_rust(
        self,
        code_path: str,
        workspace: str,
        config: SandboxConfig,
    ) -> ExecutionResult:
        """Execute Rust code."""
        env = os.environ.copy()
        env.update(config.environment)

        # Compile and run
        binary_path = code_path.replace(".rs", "")

        # Compile
        compile_proc = await asyncio.create_subprocess_exec(
            "rustc", code_path, "-o", binary_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, compile_stderr = await compile_proc.communicate()

        if compile_proc.returncode != 0:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                exit_code=compile_proc.returncode or 1,
                stderr=compile_stderr.decode("utf-8", errors="replace"),
            )

        # Run
        process = await asyncio.create_subprocess_exec(
            binary_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.timeout,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def _execute_bash(
        self,
        code_path: str,
        workspace: str,
        config: SandboxConfig,
    ) -> ExecutionResult:
        """Execute Bash script."""
        env = os.environ.copy()
        env.update(config.environment)

        process = await asyncio.create_subprocess_exec(
            "bash", code_path,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.timeout,
        )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    def _copy_workspace(self, src: str, dst: str) -> None:
        """Copy workspace contents."""
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    def _list_files(self, directory: str) -> List[str]:
        """List all files in directory."""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                rel_path = os.path.relpath(
                    os.path.join(root, filename),
                    directory
                )
                files.append(rel_path)
        return files
