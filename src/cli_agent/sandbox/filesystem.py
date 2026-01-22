"""
Transactional Filesystem - Snapshot-based file operations.

Based on research: "Transactional Filesystem Snapshots" for safe
agent operations with 100% interception and ~14.5% overhead.

Enables:
- Create snapshots before operations
- Rollback on failure
- Atomic multi-file changes
"""

import os
import shutil
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel


@dataclass
class FileState:
    """State of a single file at snapshot time."""
    path: str
    exists: bool
    content_hash: Optional[str] = None
    size: int = 0
    modified_time: float = 0.0
    is_directory: bool = False

    @classmethod
    def from_path(cls, base_path: str, rel_path: str) -> "FileState":
        """Create FileState from actual file."""
        full_path = os.path.join(base_path, rel_path)

        if not os.path.exists(full_path):
            return cls(path=rel_path, exists=False)

        if os.path.isdir(full_path):
            stat = os.stat(full_path)
            return cls(
                path=rel_path,
                exists=True,
                is_directory=True,
                modified_time=stat.st_mtime,
            )

        stat = os.stat(full_path)
        with open(full_path, "rb") as f:
            content_hash = hashlib.md5(f.read()).hexdigest()

        return cls(
            path=rel_path,
            exists=True,
            content_hash=content_hash,
            size=stat.st_size,
            modified_time=stat.st_mtime,
        )


@dataclass
class ChangeSet:
    """Set of changes made since a snapshot."""
    created: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.created or self.modified or self.deleted)

    @property
    def total_changes(self) -> int:
        """Total number of changed files."""
        return len(self.created) + len(self.modified) + len(self.deleted)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "created": self.created,
            "modified": self.modified,
            "deleted": self.deleted,
            "total": self.total_changes,
        }


@dataclass
class Snapshot:
    """
    Filesystem snapshot at a point in time.

    Captures file states for comparison and rollback.
    """
    id: str
    workspace: str
    created_at: datetime
    file_states: Dict[str, FileState] = field(default_factory=dict)
    backup_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_changes(self, current_workspace: Optional[str] = None) -> ChangeSet:
        """
        Compare snapshot to current state.

        Args:
            current_workspace: Path to compare (defaults to original)

        Returns:
            ChangeSet with differences
        """
        workspace = current_workspace or self.workspace
        changes = ChangeSet()

        # Get current files
        current_files: Set[str] = set()
        for root, dirs, files in os.walk(workspace):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in files:
                if filename.startswith("."):
                    continue
                rel_path = os.path.relpath(
                    os.path.join(root, filename),
                    workspace
                )
                current_files.add(rel_path)

                # Check if file exists in snapshot
                if rel_path not in self.file_states:
                    changes.created.append(rel_path)
                else:
                    # Check if modified
                    old_state = self.file_states[rel_path]
                    new_state = FileState.from_path(workspace, rel_path)

                    if new_state.content_hash != old_state.content_hash:
                        changes.modified.append(rel_path)

        # Check for deleted files
        for path in self.file_states:
            if path not in current_files and self.file_states[path].exists:
                if not self.file_states[path].is_directory:
                    changes.deleted.append(path)

        return changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workspace": self.workspace,
            "created_at": self.created_at.isoformat(),
            "file_count": len(self.file_states),
            "has_backup": self.backup_path is not None,
            "metadata": self.metadata,
        }


class TransactionalFS:
    """
    Transactional filesystem with snapshot support.

    Provides:
    - Snapshot creation before operations
    - Rollback on failure
    - Change tracking
    - Atomic multi-file operations
    """

    def __init__(
        self,
        workspace: str,
        backup_dir: Optional[str] = None,
        max_snapshots: int = 10,
    ):
        self.workspace = workspace
        self.backup_dir = backup_dir or os.path.join(workspace, ".snapshots")
        self.max_snapshots = max_snapshots

        self._snapshots: Dict[str, Snapshot] = {}
        self._snapshot_counter = 0
        self._active_transaction: Optional[str] = None

        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_snapshot(
        self,
        name: Optional[str] = None,
        backup: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Snapshot:
        """
        Create a snapshot of current state.

        Args:
            name: Optional snapshot name
            backup: Whether to create physical backup
            metadata: Optional metadata to attach

        Returns:
            Snapshot instance
        """
        self._snapshot_counter += 1
        snapshot_id = name or f"snapshot_{self._snapshot_counter}"

        # Collect file states
        file_states: Dict[str, FileState] = {}

        for root, dirs, files in os.walk(self.workspace):
            # Skip backup directory and hidden dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and
                os.path.join(root, d) != self.backup_dir
            ]

            for filename in files:
                if filename.startswith("."):
                    continue

                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.workspace)
                file_states[rel_path] = FileState.from_path(self.workspace, rel_path)

        # Create backup if requested
        backup_path = None
        if backup:
            backup_path = os.path.join(self.backup_dir, snapshot_id)
            self._create_backup(backup_path, file_states.keys())

        snapshot = Snapshot(
            id=snapshot_id,
            workspace=self.workspace,
            created_at=datetime.now(),
            file_states=file_states,
            backup_path=backup_path,
            metadata=metadata or {},
        )

        self._snapshots[snapshot_id] = snapshot

        # Limit snapshot count
        self._cleanup_old_snapshots()

        return snapshot

    def rollback(self, snapshot: Snapshot) -> ChangeSet:
        """
        Rollback to a snapshot state.

        Args:
            snapshot: Snapshot to rollback to

        Returns:
            ChangeSet of changes reverted
        """
        if not snapshot.backup_path or not os.path.exists(snapshot.backup_path):
            raise ValueError("Snapshot has no backup to restore from")

        # Get current changes before rollback
        changes = snapshot.get_changes()

        # Remove created files
        for path in changes.created:
            full_path = os.path.join(self.workspace, path)
            if os.path.exists(full_path):
                os.remove(full_path)

        # Restore modified and deleted files from backup
        for path in changes.modified + changes.deleted:
            backup_file = os.path.join(snapshot.backup_path, path)
            target_file = os.path.join(self.workspace, path)

            if os.path.exists(backup_file):
                # Ensure directory exists
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copy2(backup_file, target_file)

        return changes

    def begin_transaction(self, name: Optional[str] = None) -> Snapshot:
        """
        Begin a transaction with automatic snapshot.

        Args:
            name: Optional transaction name

        Returns:
            Snapshot created for the transaction
        """
        if self._active_transaction:
            raise RuntimeError("Transaction already in progress")

        snapshot = self.create_snapshot(
            name=name or "transaction",
            backup=True,
            metadata={"type": "transaction"},
        )

        self._active_transaction = snapshot.id
        return snapshot

    def commit_transaction(self) -> ChangeSet:
        """
        Commit the current transaction.

        Returns:
            ChangeSet of committed changes
        """
        if not self._active_transaction:
            raise RuntimeError("No transaction in progress")

        snapshot = self._snapshots.get(self._active_transaction)
        if not snapshot:
            raise RuntimeError("Transaction snapshot not found")

        changes = snapshot.get_changes()
        self._active_transaction = None

        return changes

    def rollback_transaction(self) -> ChangeSet:
        """
        Rollback the current transaction.

        Returns:
            ChangeSet of reverted changes
        """
        if not self._active_transaction:
            raise RuntimeError("No transaction in progress")

        snapshot = self._snapshots.get(self._active_transaction)
        if not snapshot:
            raise RuntimeError("Transaction snapshot not found")

        changes = self.rollback(snapshot)
        self._active_transaction = None

        return changes

    def get_changes_since(self, snapshot_id: str) -> ChangeSet:
        """
        Get changes since a snapshot.

        Args:
            snapshot_id: ID of snapshot to compare

        Returns:
            ChangeSet since snapshot
        """
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        return snapshot.get_changes()

    def get_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """Get a snapshot by ID."""
        return self._snapshots.get(snapshot_id)

    def list_snapshots(self) -> List[Snapshot]:
        """List all snapshots."""
        return list(self._snapshots.values())

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot and its backup.

        Returns:
            True if snapshot was deleted
        """
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            return False

        # Remove backup
        if snapshot.backup_path and os.path.exists(snapshot.backup_path):
            shutil.rmtree(snapshot.backup_path, ignore_errors=True)

        del self._snapshots[snapshot_id]
        return True

    def _create_backup(self, backup_path: str, files: List[str]) -> None:
        """Create physical backup of files."""
        os.makedirs(backup_path, exist_ok=True)

        for rel_path in files:
            src = os.path.join(self.workspace, rel_path)
            dst = os.path.join(backup_path, rel_path)

            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond limit."""
        if len(self._snapshots) <= self.max_snapshots:
            return

        # Sort by creation time
        sorted_snapshots = sorted(
            self._snapshots.values(),
            key=lambda s: s.created_at,
        )

        # Remove oldest (skip active transaction)
        to_remove = len(sorted_snapshots) - self.max_snapshots
        for snapshot in sorted_snapshots[:to_remove]:
            if snapshot.id != self._active_transaction:
                self.delete_snapshot(snapshot.id)

    def save_metadata(self, path: Optional[str] = None) -> None:
        """Save snapshot metadata to file."""
        path = path or os.path.join(self.backup_dir, "snapshots.json")

        data = {
            "workspace": self.workspace,
            "snapshots": [s.to_dict() for s in self._snapshots.values()],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_metadata(self, path: Optional[str] = None) -> None:
        """Load snapshot metadata from file."""
        path = path or os.path.join(self.backup_dir, "snapshots.json")

        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct snapshots (file states would need to be rebuilt)
        for snap_data in data.get("snapshots", []):
            if snap_data["id"] not in self._snapshots:
                # Snapshot metadata only - file states need rebuilding if needed
                snapshot = Snapshot(
                    id=snap_data["id"],
                    workspace=snap_data["workspace"],
                    created_at=datetime.fromisoformat(snap_data["created_at"]),
                    backup_path=os.path.join(self.backup_dir, snap_data["id"]),
                    metadata=snap_data.get("metadata", {}),
                )
                self._snapshots[snap_data["id"]] = snapshot
