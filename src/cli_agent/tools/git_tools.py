"""
Git Tools - Deterministic git operations with Pydantic validation.

All git tools are:
- Type-safe with Pydantic schemas
- Safe (no force operations without explicit confirmation)
- Well-documented for LLM tool use
- Support worktrees for parallel agent work
"""

import os
import subprocess
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class GitStatusInput(BaseModel):
    """Input schema for git status operation."""
    path: str = Field(default=".", description="Repository path")
    include_untracked: bool = Field(default=True)


class GitStatusOutput(BaseModel):
    """Output schema for git status operation."""
    success: bool
    branch: str = ""
    staged: List[str] = []
    modified: List[str] = []
    untracked: List[str] = []
    ahead: int = 0
    behind: int = 0
    clean: bool = True
    error: Optional[str] = None


class GitStatusTool:
    """
    Get git repository status.

    Returns current branch, staged/modified/untracked files,
    and sync status with remote.
    """

    name: str = "git_status"
    description: str = "Get the status of a git repository"
    input_schema = GitStatusInput
    output_schema = GitStatusOutput

    def execute(self, input_data: GitStatusInput) -> GitStatusOutput:
        """Execute git status operation."""
        try:
            # Get current branch
            branch = self._run_git(
                ["rev-parse", "--abbrev-ref", "HEAD"],
                input_data.path
            ).strip()

            # Get status
            status_output = self._run_git(
                ["status", "--porcelain"],
                input_data.path
            )

            staged = []
            modified = []
            untracked = []

            for line in status_output.split("\n"):
                if not line:
                    continue
                status_code = line[:2]
                file_path = line[3:]

                if status_code[0] in "MADRC":
                    staged.append(file_path)
                if status_code[1] in "MD":
                    modified.append(file_path)
                if status_code == "??":
                    if input_data.include_untracked:
                        untracked.append(file_path)

            # Get ahead/behind
            ahead, behind = self._get_ahead_behind(input_data.path, branch)

            clean = not (staged or modified or untracked)

            return GitStatusOutput(
                success=True,
                branch=branch,
                staged=staged,
                modified=modified,
                untracked=untracked,
                ahead=ahead,
                behind=behind,
                clean=clean,
            )

        except Exception as e:
            return GitStatusOutput(
                success=False,
                error=str(e),
            )

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run git command."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

    def _get_ahead_behind(self, path: str, branch: str) -> tuple:
        """Get ahead/behind counts."""
        try:
            output = self._run_git(
                ["rev-list", "--left-right", "--count", f"origin/{branch}...{branch}"],
                path
            )
            parts = output.strip().split()
            return int(parts[1]), int(parts[0])
        except Exception:
            return 0, 0


class GitDiffInput(BaseModel):
    """Input schema for git diff operation."""
    path: str = Field(default=".", description="Repository path")
    file_path: Optional[str] = Field(default=None, description="Specific file to diff")
    staged: bool = Field(default=False, description="Show staged changes")
    commit: Optional[str] = Field(default=None, description="Compare against commit")


class GitDiffOutput(BaseModel):
    """Output schema for git diff operation."""
    success: bool
    diff: str = ""
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    error: Optional[str] = None


class GitDiffTool:
    """
    Get git diff.

    Shows changes between working directory, staging area,
    or specific commits.
    """

    name: str = "git_diff"
    description: str = "Get git diff for changes"
    input_schema = GitDiffInput
    output_schema = GitDiffOutput

    def execute(self, input_data: GitDiffInput) -> GitDiffOutput:
        """Execute git diff operation."""
        try:
            args = ["diff"]

            if input_data.staged:
                args.append("--staged")

            if input_data.commit:
                args.append(input_data.commit)

            if input_data.file_path:
                args.extend(["--", input_data.file_path])

            diff = self._run_git(args, input_data.path)

            # Get stats
            stat_args = args + ["--stat"]
            stat_output = self._run_git(stat_args, input_data.path)

            files_changed = 0
            insertions = 0
            deletions = 0

            # Parse stat line (e.g., "3 files changed, 10 insertions(+), 5 deletions(-)")
            for line in stat_output.split("\n"):
                if "changed" in line:
                    import re
                    files_match = re.search(r"(\d+) files? changed", line)
                    ins_match = re.search(r"(\d+) insertions?", line)
                    del_match = re.search(r"(\d+) deletions?", line)

                    if files_match:
                        files_changed = int(files_match.group(1))
                    if ins_match:
                        insertions = int(ins_match.group(1))
                    if del_match:
                        deletions = int(del_match.group(1))

            return GitDiffOutput(
                success=True,
                diff=diff,
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
            )

        except Exception as e:
            return GitDiffOutput(
                success=False,
                error=str(e),
            )

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run git command."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout


class GitCommitInput(BaseModel):
    """Input schema for git commit operation."""
    path: str = Field(default=".", description="Repository path")
    message: str = Field(description="Commit message")
    files: List[str] = Field(default_factory=list, description="Files to stage (empty=all)")
    amend: bool = Field(default=False, description="Amend previous commit")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Commit message cannot be empty")
        return v


class GitCommitOutput(BaseModel):
    """Output schema for git commit operation."""
    success: bool
    commit_hash: str = ""
    message: str = ""
    files_committed: int = 0
    error: Optional[str] = None


class GitCommitTool:
    """
    Create a git commit.

    Stages specified files and creates commit.
    Supports amending previous commit.
    """

    name: str = "git_commit"
    description: str = "Create a git commit with specified message"
    input_schema = GitCommitInput
    output_schema = GitCommitOutput

    def execute(self, input_data: GitCommitInput) -> GitCommitOutput:
        """Execute git commit operation."""
        try:
            # Stage files
            if input_data.files:
                for file in input_data.files:
                    self._run_git(["add", file], input_data.path)
            else:
                self._run_git(["add", "-A"], input_data.path)

            # Create commit
            commit_args = ["commit", "-m", input_data.message]
            if input_data.amend:
                commit_args.append("--amend")

            self._run_git(commit_args, input_data.path)

            # Get commit info
            hash_output = self._run_git(
                ["rev-parse", "HEAD"],
                input_data.path
            ).strip()

            # Get files committed count
            show_output = self._run_git(
                ["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
                input_data.path
            )
            files_committed = len([l for l in show_output.split("\n") if l.strip()])

            return GitCommitOutput(
                success=True,
                commit_hash=hash_output,
                message=input_data.message,
                files_committed=files_committed,
            )

        except Exception as e:
            return GitCommitOutput(
                success=False,
                error=str(e),
            )

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run git command."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout


class GitBranchInput(BaseModel):
    """Input schema for git branch operation."""
    path: str = Field(default=".", description="Repository path")
    name: Optional[str] = Field(default=None, description="Branch name to create/switch")
    action: str = Field(
        default="list",
        description="Action: list, create, switch, delete"
    )
    force: bool = Field(default=False, description="Force operation")


class GitBranchOutput(BaseModel):
    """Output schema for git branch operation."""
    success: bool
    branches: List[Dict[str, Any]] = []
    current_branch: str = ""
    error: Optional[str] = None


class GitBranchTool:
    """
    Manage git branches.

    List, create, switch, or delete branches.
    """

    name: str = "git_branch"
    description: str = "Manage git branches"
    input_schema = GitBranchInput
    output_schema = GitBranchOutput

    def execute(self, input_data: GitBranchInput) -> GitBranchOutput:
        """Execute git branch operation."""
        try:
            if input_data.action == "list":
                return self._list_branches(input_data.path)
            elif input_data.action == "create":
                return self._create_branch(input_data.path, input_data.name)
            elif input_data.action == "switch":
                return self._switch_branch(input_data.path, input_data.name)
            elif input_data.action == "delete":
                return self._delete_branch(
                    input_data.path, input_data.name, input_data.force
                )
            else:
                return GitBranchOutput(
                    success=False,
                    error=f"Unknown action: {input_data.action}",
                )

        except Exception as e:
            return GitBranchOutput(
                success=False,
                error=str(e),
            )

    def _list_branches(self, path: str) -> GitBranchOutput:
        """List all branches."""
        output = self._run_git(
            ["branch", "-a", "-v"],
            path
        )

        branches = []
        current = ""

        for line in output.split("\n"):
            if not line.strip():
                continue

            is_current = line.startswith("*")
            line = line.lstrip("* ")
            parts = line.split()

            if parts:
                branch_name = parts[0]
                commit = parts[1] if len(parts) > 1 else ""

                branches.append({
                    "name": branch_name,
                    "commit": commit,
                    "current": is_current,
                })

                if is_current:
                    current = branch_name

        return GitBranchOutput(
            success=True,
            branches=branches,
            current_branch=current,
        )

    def _create_branch(self, path: str, name: str) -> GitBranchOutput:
        """Create a new branch."""
        self._run_git(["checkout", "-b", name], path)
        return GitBranchOutput(
            success=True,
            current_branch=name,
        )

    def _switch_branch(self, path: str, name: str) -> GitBranchOutput:
        """Switch to a branch."""
        self._run_git(["checkout", name], path)
        return GitBranchOutput(
            success=True,
            current_branch=name,
        )

    def _delete_branch(self, path: str, name: str, force: bool) -> GitBranchOutput:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        self._run_git(["branch", flag, name], path)
        return self._list_branches(path)

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run git command."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout


class GitLogInput(BaseModel):
    """Input schema for git log operation."""
    path: str = Field(default=".", description="Repository path")
    count: int = Field(default=10, ge=1, le=100, description="Number of commits")
    file_path: Optional[str] = Field(default=None, description="Filter by file")
    since: Optional[str] = Field(default=None, description="Since date")
    author: Optional[str] = Field(default=None, description="Filter by author")


class GitLogOutput(BaseModel):
    """Output schema for git log operation."""
    success: bool
    commits: List[Dict[str, Any]] = []
    error: Optional[str] = None


class GitLogTool:
    """
    Get git commit history.

    Returns recent commits with hash, author, date, and message.
    Supports filtering by file, date, and author.
    """

    name: str = "git_log"
    description: str = "Get git commit history"
    input_schema = GitLogInput
    output_schema = GitLogOutput

    def execute(self, input_data: GitLogInput) -> GitLogOutput:
        """Execute git log operation."""
        try:
            args = [
                "log",
                f"-{input_data.count}",
                "--pretty=format:%H|%an|%ae|%ai|%s",
            ]

            if input_data.since:
                args.append(f"--since={input_data.since}")
            if input_data.author:
                args.append(f"--author={input_data.author}")
            if input_data.file_path:
                args.extend(["--", input_data.file_path])

            output = self._run_git(args, input_data.path)

            commits = []
            for line in output.split("\n"):
                if not line.strip():
                    continue
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4],
                    })

            return GitLogOutput(
                success=True,
                commits=commits,
            )

        except Exception as e:
            return GitLogOutput(
                success=False,
                error=str(e),
            )

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run git command."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout


class GitWorktreeInput(BaseModel):
    """Input schema for git worktree operation."""
    path: str = Field(default=".", description="Repository path")
    action: str = Field(description="Action: list, add, remove")
    worktree_path: Optional[str] = Field(default=None, description="Worktree path")
    branch: Optional[str] = Field(default=None, description="Branch for new worktree")


class GitWorktreeOutput(BaseModel):
    """Output schema for git worktree operation."""
    success: bool
    worktrees: List[Dict[str, str]] = []
    created_path: Optional[str] = None
    error: Optional[str] = None


class GitWorktreeTool:
    """
    Manage git worktrees.

    Worktrees allow parallel agent work on different features
    in separate directories.
    """

    name: str = "git_worktree"
    description: str = "Manage git worktrees for parallel work"
    input_schema = GitWorktreeInput
    output_schema = GitWorktreeOutput

    def execute(self, input_data: GitWorktreeInput) -> GitWorktreeOutput:
        """Execute git worktree operation."""
        try:
            if input_data.action == "list":
                return self._list_worktrees(input_data.path)
            elif input_data.action == "add":
                return self._add_worktree(
                    input_data.path,
                    input_data.worktree_path,
                    input_data.branch,
                )
            elif input_data.action == "remove":
                return self._remove_worktree(
                    input_data.path,
                    input_data.worktree_path,
                )
            else:
                return GitWorktreeOutput(
                    success=False,
                    error=f"Unknown action: {input_data.action}",
                )

        except Exception as e:
            return GitWorktreeOutput(
                success=False,
                error=str(e),
            )

    def _list_worktrees(self, path: str) -> GitWorktreeOutput:
        """List all worktrees."""
        output = self._run_git(["worktree", "list", "--porcelain"], path)

        worktrees = []
        current = {}

        for line in output.split("\n"):
            if line.startswith("worktree "):
                if current:
                    worktrees.append(current)
                current = {"path": line[9:]}
            elif line.startswith("HEAD "):
                current["head"] = line[5:]
            elif line.startswith("branch "):
                current["branch"] = line[7:]

        if current:
            worktrees.append(current)

        return GitWorktreeOutput(
            success=True,
            worktrees=worktrees,
        )

    def _add_worktree(
        self,
        path: str,
        worktree_path: str,
        branch: Optional[str],
    ) -> GitWorktreeOutput:
        """Add a new worktree."""
        args = ["worktree", "add", worktree_path]
        if branch:
            args.extend(["-b", branch])

        self._run_git(args, path)

        return GitWorktreeOutput(
            success=True,
            created_path=worktree_path,
        )

    def _remove_worktree(self, path: str, worktree_path: str) -> GitWorktreeOutput:
        """Remove a worktree."""
        self._run_git(["worktree", "remove", worktree_path], path)
        return self._list_worktrees(path)

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run git command."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout
