"""
Cognitive Core — Spec-Locking Manifest (S-008)

At instance start, captures the exact YAML configuration in effect:
workflow YAML, domain YAML, coordinator config, and prompt templates.
Stores both SHA-256 hashes and full content for reconstruction.

On/off toggle: CC_MANIFEST_ENABLED env var or manifest.enabled in config.

Usage:
    manifest = SpecManifest.capture(
        workflow_path="workflows/product_return.yaml",
        domain_path="domains/electronics_return.yaml",
        coordinator_path="coordinator/config.yaml",
        prompt_dir="registry/prompts",
    )

    # Store in audit trail
    audit.record_manifest(trace_id, manifest.to_dict())

    # Later — verify configs haven't changed
    drifts = manifest.verify()
    # [{"file": "domains/electronics_return.yaml", "stored_hash": "abc...", "current_hash": "def..."}]
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _hash_content(content: str) -> str:
    """SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_file(path: str) -> str | None:
    """Read file content, return None if not found."""
    try:
        with open(path, "r") as f:
            return f.read()
    except (FileNotFoundError, PermissionError):
        return None


@dataclass
class FileSnapshot:
    """Snapshot of a single file."""
    path: str
    hash: str
    content: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "hash": self.hash,
            "size_bytes": self.size_bytes,
            "content": self.content,
        }

    @staticmethod
    def from_dict(d: dict) -> FileSnapshot:
        return FileSnapshot(
            path=d["path"],
            hash=d["hash"],
            content=d.get("content", ""),
            size_bytes=d.get("size_bytes", 0),
        )


@dataclass
class SpecManifest:
    """
    Immutable snapshot of all configuration in effect at instance start.
    """
    manifest_hash: str = ""
    captured_at: float = 0.0
    files: dict[str, FileSnapshot] = field(default_factory=dict)

    @staticmethod
    def capture(
        workflow_path: str = "",
        domain_path: str = "",
        coordinator_path: str = "coordinator/config.yaml",
        prompt_dir: str = "registry/prompts",
        include_content: bool = True,
    ) -> SpecManifest:
        """
        Capture a manifest from current file system state.

        Args:
            workflow_path: Path to workflow YAML
            domain_path: Path to domain YAML
            coordinator_path: Path to coordinator config
            prompt_dir: Directory containing prompt templates
            include_content: Whether to store full file content (True for Option B)
        """
        files: dict[str, FileSnapshot] = {}

        # Core config files
        for label, path in [
            ("workflow", workflow_path),
            ("domain", domain_path),
            ("coordinator", coordinator_path),
        ]:
            if path:
                content = _read_file(path)
                if content is not None:
                    files[label] = FileSnapshot(
                        path=path,
                        hash=_hash_content(content),
                        content=content if include_content else "",
                        size_bytes=len(content.encode("utf-8")),
                    )

        # Prompt templates
        if prompt_dir and os.path.isdir(prompt_dir):
            for fname in sorted(os.listdir(prompt_dir)):
                if fname.endswith(".txt"):
                    fpath = os.path.join(prompt_dir, fname)
                    content = _read_file(fpath)
                    if content is not None:
                        label = f"prompt:{fname}"
                        files[label] = FileSnapshot(
                            path=fpath,
                            hash=_hash_content(content),
                            content=content if include_content else "",
                            size_bytes=len(content.encode("utf-8")),
                        )

        # Compute manifest hash from all file hashes
        combined = "|".join(
            f"{k}:{v.hash}" for k, v in sorted(files.items())
        )
        manifest_hash = _hash_content(combined)

        return SpecManifest(
            manifest_hash=manifest_hash,
            captured_at=time.time(),
            files=files,
        )

    def verify(self) -> list[dict[str, str]]:
        """
        Verify current files against manifest.
        Returns list of drifted files (empty = no drift).
        """
        drifts = []
        for label, snapshot in self.files.items():
            current_content = _read_file(snapshot.path)
            if current_content is None:
                drifts.append({
                    "label": label,
                    "file": snapshot.path,
                    "status": "missing",
                    "stored_hash": snapshot.hash,
                    "current_hash": "",
                })
            else:
                current_hash = _hash_content(current_content)
                if current_hash != snapshot.hash:
                    drifts.append({
                        "label": label,
                        "file": snapshot.path,
                        "status": "changed",
                        "stored_hash": snapshot.hash,
                        "current_hash": current_hash,
                    })
        return drifts

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage in audit trail."""
        return {
            "manifest_hash": self.manifest_hash,
            "captured_at": self.captured_at,
            "file_count": len(self.files),
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> SpecManifest:
        """Reconstruct from stored dict."""
        files = {}
        for k, v in d.get("files", {}).items():
            files[k] = FileSnapshot.from_dict(v)
        return SpecManifest(
            manifest_hash=d.get("manifest_hash", ""),
            captured_at=d.get("captured_at", 0.0),
            files=files,
        )

    def get_file_content(self, label: str) -> str | None:
        """Retrieve stored content for a specific file."""
        snap = self.files.get(label)
        return snap.content if snap else None

    @property
    def file_labels(self) -> list[str]:
        return sorted(self.files.keys())


def is_manifest_enabled() -> bool:
    """Check if manifest capture is enabled."""
    val = os.environ.get("CC_MANIFEST_ENABLED", "true").lower()
    return val in ("true", "1", "yes", "on")
