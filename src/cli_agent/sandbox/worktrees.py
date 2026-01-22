"""
Git Worktree Manager - Parallel Agent Work

Based on research: "Git Worktrees" for parallel agent work
on different features simultaneously.

Enables:
- Multiple working directories for same repo
- Parallel feature development
- Isolated agent workspaces
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Worktree:
    """
    A git worktree for isolated work.

    Represents a separate working directory linked
    to the same git repository.
    """
    path: str
    branch: str
    head_commit: str = ""
    is_main: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get worktree name from path."""
        return os.path.basename(self.path)

    @property
    def exists(self) -> bool:
        """Check if worktree directory exists."""
        return os.path.exists(self.path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "branch": self.branch,
            "head_commit": self.head_commit,
            "is_main": self.is_main,
            "name": self.name,
            "exists": self.exists,
        }


class WorktreeManager:
    """
    Manages git worktrees for parallel agent work.

    Features:
    - Create worktrees for agents
    - Switch between worktrees
    - Merge worktree changes
    - Clean up completed worktrees
    """

    def __init__(
        self,
        repo_path: str,
        worktree_base: Optional[str] = None,
    ):
        """
        Initialize worktree manager.

        Args:
            repo_path: Path to main git repository
            worktree_base: Base directory for worktrees
        """
        self.repo_path = os.path.abspath(repo_path)
        self.worktree_base = worktree_base or os.path.join(
            os.path.dirname(self.repo_path),
            ".worktrees"
        )

        self._worktrees: Dict[str, Worktree] = {}

        # Ensure worktree base exists
        os.makedirs(self.worktree_base, exist_ok=True)

        # Load existing worktrees
        self._load_worktrees()

    def create_worktree(
        self,
        name: str,
        branch: Optional[str] = None,
        base_branch: str = "main",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Worktree:
        """
        Create a new worktree.

        Args:
            name: Worktree name
            branch: Branch name (defaults to name)
            base_branch: Branch to create from
            metadata: Optional metadata

        Returns:
            Created Worktree instance
        """
        branch = branch or name
        worktree_path = os.path.join(self.worktree_base, name)

        if os.path.exists(worktree_path):
            raise ValueError(f"Worktree path already exists: {worktree_path}")

        # Check if branch exists
        branch_exists = self._branch_exists(branch)

        if branch_exists:
            # Use existing branch
            self._run_git(["worktree", "add", worktree_path, branch])
        else:
            # Create new branch from base
            self._run_git([
                "worktree", "add", "-b", branch, worktree_path, base_branch
            ])

        # Get head commit
        head_commit = self._get_head_commit(worktree_path)

        worktree = Worktree(
            path=worktree_path,
            branch=branch,
            head_commit=head_commit,
            metadata=metadata or {},
        )

        self._worktrees[name] = worktree
        return worktree

    def remove_worktree(
        self,
        name: str,
        force: bool = False,
        delete_branch: bool = False,
    ) -> bool:
        """
        Remove a worktree.

        Args:
            name: Worktree name
            force: Force removal even if dirty
            delete_branch: Also delete the branch

        Returns:
            True if removed successfully
        """
        worktree = self._worktrees.get(name)
        if not worktree:
            return False

        # Remove worktree
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(worktree.path)

        try:
            self._run_git(args)
        except Exception:
            if force:
                # Force directory removal
                import shutil
                if os.path.exists(worktree.path):
                    shutil.rmtree(worktree.path, ignore_errors=True)

        # Delete branch if requested
        if delete_branch and not worktree.is_main:
            try:
                flag = "-D" if force else "-d"
                self._run_git(["branch", flag, worktree.branch])
            except Exception:
                pass  # Branch might not exist or be protected

        del self._worktrees[name]
        return True

    def get_worktree(self, name: str) -> Optional[Worktree]:
        """Get a worktree by name."""
        return self._worktrees.get(name)

    def list_worktrees(self) -> List[Worktree]:
        """List all worktrees."""
        self._load_worktrees()  # Refresh
        return list(self._worktrees.values())

    def get_worktree_for_task(
        self,
        task_id: str,
        base_branch: str = "main",
    ) -> Worktree:
        """
        Get or create a worktree for a task.

        Args:
            task_id: Task identifier
            base_branch: Branch to create from if new

        Returns:
            Worktree for the task
        """
        name = f"task_{task_id}"

        if name in self._worktrees:
            return self._worktrees[name]

        return self.create_worktree(
            name=name,
            branch=f"feature/{task_id}",
            base_branch=base_branch,
            metadata={"task_id": task_id},
        )

    def merge_worktree(
        self,
        name: str,
        target_branch: str = "main",
        squash: bool = False,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Merge worktree changes to target branch.

        Args:
            name: Worktree name
            target_branch: Branch to merge into
            squash: Squash commits
            message: Custom merge message

        Returns:
            Merge result info
        """
        worktree = self._worktrees.get(name)
        if not worktree:
            raise ValueError(f"Worktree not found: {name}")

        # Get commits to merge
        commits = self._get_commits_ahead(worktree.branch, target_branch)

        if not commits:
            return {
                "merged": False,
                "reason": "No commits to merge",
                "commits": [],
            }

        # Switch to target branch in main repo
        original_branch = self._get_current_branch()
        self._run_git(["checkout", target_branch])

        try:
            # Merge
            merge_args = ["merge", worktree.branch]
            if squash:
                merge_args.append("--squash")
            if message:
                merge_args.extend(["-m", message])

            self._run_git(merge_args)

            if squash:
                # Need to commit after squash
                commit_msg = message or f"Squashed commits from {worktree.branch}"
                self._run_git(["commit", "-m", commit_msg])

            return {
                "merged": True,
                "commits": commits,
                "squashed": squash,
                "target_branch": target_branch,
            }

        finally:
            # Restore original branch
            if original_branch:
                self._run_git(["checkout", original_branch])

    def sync_worktree(
        self,
        name: str,
        from_branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Sync worktree with another branch (rebase or merge).

        Args:
            name: Worktree name
            from_branch: Branch to sync from

        Returns:
            Sync result info
        """
        worktree = self._worktrees.get(name)
        if not worktree:
            raise ValueError(f"Worktree not found: {name}")

        # Run rebase in worktree
        try:
            result = subprocess.run(
                ["git", "rebase", from_branch],
                cwd=worktree.path,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return {
                    "synced": True,
                    "method": "rebase",
                    "from_branch": from_branch,
                }
            else:
                # Abort rebase and try merge
                subprocess.run(
                    ["git", "rebase", "--abort"],
                    cwd=worktree.path,
                    capture_output=True,
                )

                merge_result = subprocess.run(
                    ["git", "merge", from_branch],
                    cwd=worktree.path,
                    capture_output=True,
                    text=True,
                )

                return {
                    "synced": merge_result.returncode == 0,
                    "method": "merge",
                    "from_branch": from_branch,
                    "error": merge_result.stderr if merge_result.returncode != 0 else None,
                }

        except Exception as e:
            return {
                "synced": False,
                "error": str(e),
            }

    def get_worktree_status(self, name: str) -> Dict[str, Any]:
        """
        Get detailed status of a worktree.

        Args:
            name: Worktree name

        Returns:
            Status information
        """
        worktree = self._worktrees.get(name)
        if not worktree:
            raise ValueError(f"Worktree not found: {name}")

        if not worktree.exists:
            return {
                "exists": False,
                "error": "Worktree directory not found",
            }

        # Get status in worktree
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree.path,
            capture_output=True,
            text=True,
        )

        # Parse status
        changes = {
            "staged": [],
            "modified": [],
            "untracked": [],
        }

        for line in status_result.stdout.split("\n"):
            if not line:
                continue
            status = line[:2]
            path = line[3:]

            if status[0] in "MADRC":
                changes["staged"].append(path)
            if status[1] in "MD":
                changes["modified"].append(path)
            if status == "??":
                changes["untracked"].append(path)

        # Get commits ahead of main
        commits_ahead = self._get_commits_ahead(worktree.branch, "main")

        return {
            "exists": True,
            "branch": worktree.branch,
            "head": self._get_head_commit(worktree.path),
            "clean": not any(changes.values()),
            "changes": changes,
            "commits_ahead": len(commits_ahead),
        }

    def _load_worktrees(self) -> None:
        """Load existing worktrees from git."""
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return

            current_worktree: Dict[str, str] = {}

            for line in result.stdout.split("\n"):
                if line.startswith("worktree "):
                    if current_worktree:
                        self._add_worktree_from_git(current_worktree)
                    current_worktree = {"path": line[9:]}
                elif line.startswith("HEAD "):
                    current_worktree["head"] = line[5:]
                elif line.startswith("branch "):
                    current_worktree["branch"] = line[7:].replace("refs/heads/", "")

            if current_worktree:
                self._add_worktree_from_git(current_worktree)

        except Exception:
            pass

    def _add_worktree_from_git(self, info: Dict[str, str]) -> None:
        """Add worktree from git info."""
        path = info.get("path", "")
        if not path:
            return

        name = os.path.basename(path)
        is_main = path == self.repo_path

        if name not in self._worktrees:
            self._worktrees[name] = Worktree(
                path=path,
                branch=info.get("branch", ""),
                head_commit=info.get("head", ""),
                is_main=is_main,
            )
        else:
            # Update existing
            self._worktrees[name].head_commit = info.get("head", "")

    def _run_git(self, args: List[str]) -> str:
        """Run git command in main repo."""
        result = subprocess.run(
            ["git"] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

    def _branch_exists(self, branch: str) -> bool:
        """Check if branch exists."""
        try:
            self._run_git(["rev-parse", "--verify", branch])
            return True
        except Exception:
            return False

    def _get_head_commit(self, path: str) -> str:
        """Get HEAD commit hash for path."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _get_current_branch(self) -> str:
        """Get current branch in main repo."""
        try:
            return self._run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        except Exception:
            return ""

    def _get_commits_ahead(self, branch: str, base: str) -> List[str]:
        """Get commits in branch that are not in base."""
        try:
            result = self._run_git([
                "log", "--oneline", f"{base}..{branch}"
            ])
            return [line for line in result.strip().split("\n") if line]
        except Exception:
            return []
