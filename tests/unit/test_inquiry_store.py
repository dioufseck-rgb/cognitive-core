"""Single-writer lock regressions, including the Windows mandatory-lock path."""

from contextlib import contextmanager
import errno
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from cognitive_core.inquiry import store as store_module
from cognitive_core.inquiry.store import InquiryStore


class ProtectedLockHandle(io.BytesIO):
    """A Windows lock handle whose byte contents must never be accessed."""

    def fileno(self):
        return 42

    def read(self, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Another handle owns this byte")

    def write(self, *args, **kwargs):
        raise PermissionError(errno.EACCES, "Another handle owns this byte")


class InquiryStoreLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = InquiryStore(self.tmp.name)

    @contextmanager
    def windows_backend(self, locking):
        handle = ProtectedLockHandle()
        api = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
        # Patch this module's os binding, not the global os.name; pathlib and
        # subprocess must continue to see the actual host platform.
        with patch.object(store_module, "os", SimpleNamespace(name="nt")), \
                patch.dict(sys.modules, {"msvcrt": api}), \
                patch.object(Path, "open", return_value=handle):
            yield handle, api

    def test_windows_lock_does_not_access_protected_bytes(self):
        calls = []
        with self.windows_backend(
            lambda fd, mode, length: calls.append((fd, mode, length, handle.tell()))
        ) as (handle, api):
            with self.store.locked():
                self.assertEqual(calls, [(42, api.LK_NBLCK, 1, 0)])
            self.assertEqual(calls, [(42, api.LK_NBLCK, 1, 0),
                                     (42, api.LK_UNLCK, 1, 0)])
            self.assertTrue(handle.closed)

    def test_windows_lock_conflict_reports_running_inquiry(self):
        conflict = PermissionError(errno.EACCES, "Lock already held")
        locking = Mock(side_effect=conflict)
        with self.windows_backend(locking) as (handle, api):
            with self.assertRaisesRegex(RuntimeError, "already running") as caught:
                with self.store.locked():
                    self.fail("A competing writer must not enter the lock")
            self.assertIs(caught.exception.__cause__, conflict)
            # A contender must not unlock a lock it failed to acquire.
            locking.assert_called_once_with(42, api.LK_NBLCK, 1)
            self.assertTrue(handle.closed)

    def test_windows_lock_releases_after_body_error(self):
        calls = []
        with self.windows_backend(
            lambda fd, mode, length: calls.append((fd, mode, length, handle.tell()))
        ) as (handle, api):
            with self.assertRaisesRegex(ValueError, "interrupted inquiry"):
                with self.store.locked():
                    handle.seek(8)
                    raise ValueError("interrupted inquiry")
            self.assertEqual(calls, [(42, api.LK_NBLCK, 1, 0),
                                     (42, api.LK_UNLCK, 1, 0)])
            self.assertTrue(handle.closed)

    def test_native_lock_excludes_another_process_then_releases(self):
        script = """
import sys
from cognitive_core.inquiry.store import InquiryStore

try:
    with InquiryStore(sys.argv[1]).locked():
        print("acquired")
except RuntimeError as exc:
    print(str(exc))
    sys.exit(23)
"""

        def contender():
            return subprocess.run(
                [sys.executable, "-B", "-c", script, str(self.store.directory)],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True, text=True, timeout=30,
            )

        with self.store.locked():
            self.assertEqual((self.store.directory / ".run.lock").stat().st_size, 0)
            blocked = contender()
            self.assertEqual(blocked.returncode, 23, blocked.stdout + blocked.stderr)
            self.assertIn("already running", blocked.stdout)
        acquired = contender()
        self.assertEqual(acquired.returncode, 0, acquired.stdout + acquired.stderr)
        self.assertEqual(acquired.stdout.strip(), "acquired")


if __name__ == "__main__":
    unittest.main()
