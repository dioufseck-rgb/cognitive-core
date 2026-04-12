"""
Cognitive Core — Spec-Locking Manifest (S-008)

At instance start, captures the exact config that will govern execution:
  - SHA-256 hash of workflow YAML, domain YAML, coordinator config
  - Optionally stores the full YAML content (for reconstruction without git)
  - Manifest hash stored in audit trail alongside first event

Enables regulatory reconstruction: "show me the exact logic
that was in effect when decision X was made."

Configuration:
    CC_SPEC_LOCK_ENABLED    — enable/disable (default: true)
    CC_SPEC_LOCK_SNAPSHOTS  — store full YAML content (default: true)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("cognitive_core.spec_lock")


@dataclass
class SpecManifest:
    """
    Immutable record of the configuration state at instance start.
    """
    manifest_hash: str                    # SHA-256 of all file hashes combined
    created_at: float
    file_hashes: dict[str, str]           # filepath → SHA-256
    file_contents: dict[str, str] | None  # filepath → YAML content (if snapshots enabled)
    prompt_hashes: dict[str, str]         # prompt name → SHA-256

    def to_dict(self) -> dict[str, Any]:
        result = {
            "manifest_hash": self.manifest_hash,
            "created_at": self.created_at,
            "file_hashes": self.file_hashes,
            "prompt_hashes": self.prompt_hashes,
        }
        if self.file_contents is not None:
            result["has_snapshots"] = True
            result["snapshot_size_bytes"] = sum(len(c) for c in self.file_contents.values())
        else:
            result["has_snapshots"] = False
        return result

    def to_storage(self) -> dict[str, Any]:
        """Serialize for storage (compresses snapshots)."""
        data = self.to_dict()
        if self.file_contents:
            # Compress YAML snapshots
            raw = json.dumps(self.file_contents).encode("utf-8")
            compressed = gzip.compress(raw)
            data["snapshots_compressed"] = compressed.hex()
            data["snapshots_raw_bytes"] = len(raw)
            data["snapshots_compressed_bytes"] = len(compressed)
        return data

    @staticmethod
    def from_storage(data: dict[str, Any]) -> SpecManifest:
        """Deserialize from storage."""
        contents = None
        if "snapshots_compressed" in data:
            compressed = bytes.fromhex(data["snapshots_compressed"])
            raw = gzip.decompress(compressed)
            contents = json.loads(raw)

        return SpecManifest(
            manifest_hash=data["manifest_hash"],
            created_at=data["created_at"],
            file_hashes=data["file_hashes"],
            file_contents=contents,
            prompt_hashes=data.get("prompt_hashes", {}),
        )

    def verify_against_current(self, project_root: str = ".") -> dict[str, Any]:
        """
        Compare this manifest against current files on disk.
        Returns a report of what changed.
        """
        changes = []
        for filepath, stored_hash in self.file_hashes.items():
            full_path = os.path.join(project_root, filepath)
            if not os.path.exists(full_path):
                changes.append({"file": filepath, "status": "deleted"})
            else:
                current_hash = _hash_file(full_path)
                if current_hash != stored_hash:
                    changes.append({
                        "file": filepath,
                        "status": "modified",
                        "stored_hash": stored_hash[:12],
                        "current_hash": current_hash[:12],
                    })

        return {
            "manifest_hash": self.manifest_hash,
            "created_at": self.created_at,
            "files_checked": len(self.file_hashes),
            "changes_detected": len(changes),
            "matches_current": len(changes) == 0,
            "changes": changes,
        }


def _hash_file(path: str) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_string(content: str) -> str:
    """SHA-256 hash of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def create_manifest(
    workflow_path: str,
    domain_path: str,
    coordinator_config_path: str = "",
    prompt_dir: str = "",
    project_root: str = ".",
    include_snapshots: bool = True,
) -> SpecManifest:
    """
    Create a manifest from current files on disk.

    Args:
        workflow_path: Path to workflow YAML (relative to project_root)
        domain_path: Path to domain YAML (relative to project_root)
        coordinator_config_path: Path to coordinator config (optional)
        prompt_dir: Path to prompt templates directory (optional)
        project_root: Root directory for resolving paths
        include_snapshots: Whether to store full file contents
    """
    file_hashes: dict[str, str] = {}
    file_contents: dict[str, str] | None = {} if include_snapshots else None

    # Core config files
    config_files = [workflow_path, domain_path]
    if coordinator_config_path:
        config_files.append(coordinator_config_path)

    for rel_path in config_files:
        full_path = os.path.join(project_root, rel_path)
        if os.path.exists(full_path):
            file_hashes[rel_path] = _hash_file(full_path)
            if file_contents is not None:
                with open(full_path) as f:
                    file_contents[rel_path] = f.read()

    # Prompt templates
    prompt_hashes: dict[str, str] = {}
    if prompt_dir:
        prompt_path = os.path.join(project_root, prompt_dir)
        if os.path.isdir(prompt_path):
            for fname in sorted(os.listdir(prompt_path)):
                if fname.endswith(".txt"):
                    fpath = os.path.join(prompt_path, fname)
                    prompt_hashes[fname] = _hash_file(fpath)

    # Combine all hashes into a single manifest hash
    combined = json.dumps({
        "files": file_hashes,
        "prompts": prompt_hashes,
    }, sort_keys=True)
    manifest_hash = _hash_string(combined)

    manifest = SpecManifest(
        manifest_hash=manifest_hash,
        created_at=time.time(),
        file_hashes=file_hashes,
        file_contents=file_contents,
        prompt_hashes=prompt_hashes,
    )

    logger.info("Spec manifest created: %s (%d files, %d prompts)",
                manifest_hash[:12], len(file_hashes), len(prompt_hashes))
    return manifest


# ═══════════════════════════════════════════════════════════════════
# Feature Toggle
# ═══════════════════════════════════════════════════════════════════

def is_spec_lock_enabled() -> bool:
    """Check if spec-locking is enabled."""
    return os.environ.get("CC_SPEC_LOCK_ENABLED", "true").lower() in ("true", "1", "yes")


def are_snapshots_enabled() -> bool:
    """Check if full YAML snapshots are enabled."""
    return os.environ.get("CC_SPEC_LOCK_SNAPSHOTS", "true").lower() in ("true", "1", "yes")
