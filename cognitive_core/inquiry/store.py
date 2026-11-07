"""Atomic checkpoints and a single-writer lock for a local inquiry directory."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

from cognitive_core.inquiry.models import InquiryState, utc_now


class InquiryStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.state_path = self.directory / "state.json"
        self.report_path = self.directory / "report.md"

    @contextmanager
    def locked(self):
        """OS locks are released on process exit, including an interrupted run."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ".run.lock"
        with path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt

                # Windows permits byte-range locks beyond EOF. Do not read or
                # write a byte that another handle may already have locked.
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError("This inquiry is already running in another process") from exc
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("This inquiry is already running in another process") from exc
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load(self) -> InquiryState:
        return InquiryState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def save(self, state: InquiryState) -> None:
        state.updated_at = utc_now()
        self._write(self.state_path, json.dumps(state.model_dump(mode="json"), indent=2) + "\n")

    def write_report(self, text: str) -> None:
        self._write(self.report_path, text)

    def _write(self, path: Path, text: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
