"""Tests for S-013: State Replay from Checkpoints."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_mod_path = os.path.join(_base, "engine", "replay.py")
_spec = importlib.util.spec_from_file_location("engine.replay", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.replay"] = _mod
_spec.loader.exec_module(_mod)

ReplayManager = _mod.ReplayManager
Checkpoint = _mod.Checkpoint
ReplayRecord = _mod.ReplayRecord


class TestCheckpointStorage(unittest.TestCase):
    """Test saving and retrieving checkpoints."""

    def setUp(self):
        self.db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.mgr = ReplayManager(db_path=self.db.name)

    def tearDown(self):
        self.mgr.close()
        os.unlink(self.db.name)

    def test_save_checkpoint(self):
        ckpt = self.mgr.save_checkpoint(
            "trace_1", "classify", 0,
            {"steps_completed": ["classify"], "outputs": {"classify": {"category": "fraud"}}},
        )
        self.assertTrue(ckpt.checkpoint_id.startswith("ckpt_"))
        self.assertEqual(ckpt.trace_id, "trace_1")
        self.assertEqual(ckpt.step_name, "classify")
        self.assertEqual(ckpt.step_index, 0)
        self.assertGreater(ckpt.size_bytes, 0)

    def test_get_checkpoint_by_name(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"step": "classify"})
        self.mgr.save_checkpoint("t1", "investigate", 1, {"step": "investigate"})

        ckpt = self.mgr.get_checkpoint("t1", "classify")
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt.step_name, "classify")

        ckpt2 = self.mgr.get_checkpoint("t1", "investigate")
        self.assertEqual(ckpt2.state_snapshot["step"], "investigate")

    def test_get_checkpoint_by_index(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"data": "c"})
        self.mgr.save_checkpoint("t1", "investigate", 1, {"data": "i"})

        ckpt = self.mgr.get_checkpoint_by_index("t1", 1)
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt.step_name, "investigate")

    def test_get_nonexistent_checkpoint(self):
        self.assertIsNone(self.mgr.get_checkpoint("nonexistent", "classify"))
        self.assertIsNone(self.mgr.get_checkpoint_by_index("nonexistent", 0))

    def test_list_checkpoints_ordered(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        self.mgr.save_checkpoint("t1", "investigate", 1, {"s": "i"})
        self.mgr.save_checkpoint("t1", "generate", 2, {"s": "g"})

        ckpts = self.mgr.list_checkpoints("t1")
        self.assertEqual(len(ckpts), 3)
        self.assertEqual(ckpts[0].step_name, "classify")
        self.assertEqual(ckpts[1].step_name, "investigate")
        self.assertEqual(ckpts[2].step_name, "generate")

    def test_list_empty(self):
        self.assertEqual(self.mgr.list_checkpoints("nonexistent"), [])

    def test_state_snapshot_round_trip(self):
        """Complex state survives JSON round-trip."""
        state = {
            "steps_completed": ["classify", "investigate"],
            "step_outputs": {
                "classify": {"category": "fraud", "confidence": 0.92},
                "investigate": {
                    "findings": ["suspicious pattern", "velocity alert"],
                    "evidence": {"transactions": 5, "flagged": 2},
                },
            },
            "case_input": {"member_id": "M123", "amount": 500.00},
        }
        self.mgr.save_checkpoint("t1", "investigate", 1, state)
        ckpt = self.mgr.get_checkpoint("t1", "investigate")
        self.assertEqual(ckpt.state_snapshot, state)

    def test_multiple_traces_independent(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"trace": "1"})
        self.mgr.save_checkpoint("t2", "classify", 0, {"trace": "2"})

        ckpts_t1 = self.mgr.list_checkpoints("t1")
        ckpts_t2 = self.mgr.list_checkpoints("t2")
        self.assertEqual(len(ckpts_t1), 1)
        self.assertEqual(len(ckpts_t2), 1)
        self.assertEqual(ckpts_t1[0].state_snapshot["trace"], "1")
        self.assertEqual(ckpts_t2[0].state_snapshot["trace"], "2")


class TestReplayCreation(unittest.TestCase):
    """Test replay record creation."""

    def setUp(self):
        self.db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.mgr = ReplayManager(db_path=self.db.name)

    def tearDown(self):
        self.mgr.close()
        os.unlink(self.db.name)

    def test_create_replay(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        self.mgr.save_checkpoint("t1", "investigate", 1, {"s": "i"})

        replay = self.mgr.create_replay("t1", "classify")
        self.assertIsNotNone(replay)
        self.assertTrue(replay.replay_id.startswith("replay_"))
        self.assertEqual(replay.parent_trace_id, "t1")
        self.assertEqual(replay.from_step, "classify")
        self.assertEqual(replay.from_step_index, 0)
        self.assertEqual(replay.status, "created")
        self.assertIsNone(replay.override_input)

    def test_create_replay_with_override(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        override = {"complaint": "updated complaint text"}
        replay = self.mgr.create_replay("t1", "classify", override_input=override)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.override_input, override)

    def test_create_replay_no_checkpoint_returns_none(self):
        replay = self.mgr.create_replay("nonexistent", "classify")
        self.assertIsNone(replay)

    def test_update_replay_status(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        replay = self.mgr.create_replay("t1", "classify")
        self.mgr.update_replay_status(replay.replay_id, "running")
        self.mgr.update_replay_status(replay.replay_id, "completed")

        replays = self.mgr.list_replays("t1")
        self.assertEqual(len(replays), 1)
        self.assertEqual(replays[0].status, "completed")

    def test_list_replays(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        self.mgr.create_replay("t1", "classify")
        self.mgr.create_replay("t1", "classify", override_input={"x": 1})

        replays = self.mgr.list_replays("t1")
        self.assertEqual(len(replays), 2)


class TestReplayWorkflow(unittest.TestCase):
    """Test the full replay workflow: save → list → load → replay."""

    def setUp(self):
        self.db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.mgr = ReplayManager(db_path=self.db.name)

    def tearDown(self):
        self.mgr.close()
        os.unlink(self.db.name)

    def test_full_workflow(self):
        """Simulate a 4-step workflow, fail at step 3, replay from step 2."""
        # Steps 0-2 complete successfully
        self.mgr.save_checkpoint("t1", "classify", 0, {
            "steps_completed": ["classify"],
            "outputs": {"classify": {"category": "dispute"}},
            "input": {"member": "M1"},
        })
        self.mgr.save_checkpoint("t1", "investigate", 1, {
            "steps_completed": ["classify", "investigate"],
            "outputs": {
                "classify": {"category": "dispute"},
                "investigate": {"findings": ["charge not recognized"]},
            },
            "input": {"member": "M1"},
        })
        self.mgr.save_checkpoint("t1", "challenge", 2, {
            "steps_completed": ["classify", "investigate", "challenge"],
            "outputs": {
                "classify": {"category": "dispute"},
                "investigate": {"findings": ["charge not recognized"]},
                "challenge": {"verdict": "upheld"},
            },
            "input": {"member": "M1"},
        })

        # Step 3 (generate) fails — need to debug

        # 1. List checkpoints to see what's available
        ckpts = self.mgr.list_checkpoints("t1")
        self.assertEqual(len(ckpts), 3)

        # 2. Load state at investigate (before the problem)
        ckpt = self.mgr.get_checkpoint("t1", "investigate")
        state = ckpt.state_snapshot
        self.assertEqual(state["steps_completed"], ["classify", "investigate"])

        # 3. Create replay with modified input
        replay = self.mgr.create_replay(
            "t1", "investigate",
            override_input={"member": "M1", "extra_context": "merchant confirmed refund"},
        )
        self.assertIsNotNone(replay)
        self.assertEqual(replay.from_step, "investigate")
        self.assertEqual(replay.from_step_index, 1)

        # 4. Mark replay as running then completed
        self.mgr.update_replay_status(replay.replay_id, "running")
        self.mgr.update_replay_status(replay.replay_id, "completed")


class TestCheckpointCleanup(unittest.TestCase):
    """Test cleanup operations."""

    def setUp(self):
        self.db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.mgr = ReplayManager(db_path=self.db.name)

    def tearDown(self):
        self.mgr.close()
        os.unlink(self.db.name)

    def test_delete_checkpoints(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        self.mgr.save_checkpoint("t1", "investigate", 1, {"s": "i"})
        count = self.mgr.delete_checkpoints("t1")
        self.assertEqual(count, 2)
        self.assertEqual(self.mgr.list_checkpoints("t1"), [])

    def test_delete_nonexistent(self):
        self.assertEqual(self.mgr.delete_checkpoints("nonexistent"), 0)

    def test_stats(self):
        self.mgr.save_checkpoint("t1", "classify", 0, {"s": "c"})
        self.mgr.save_checkpoint("t1", "investigate", 1, {"s": "i"})
        self.mgr.create_replay("t1", "classify")
        stats = self.mgr.checkpoint_stats()
        self.assertEqual(stats["total_checkpoints"], 2)
        self.assertGreater(stats["total_bytes"], 0)
        self.assertEqual(stats["total_replays"], 1)

    def test_empty_stats(self):
        stats = self.mgr.checkpoint_stats()
        self.assertEqual(stats["total_checkpoints"], 0)
        self.assertEqual(stats["total_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
