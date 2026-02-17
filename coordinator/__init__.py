"""
Cognitive Core — Runtime Coordinator

The fourth architectural layer. Manages multi-workflow execution,
governance tiers, cross-workflow delegation (A2A), and HITL policies
through brokered asynchronous communication.

Phase 1: In-process implementation. All instances run in the same process.
State is persisted to SQLite for suspend/resume across restarts.

Usage:
    from coordinator.runtime import Coordinator

    coord = Coordinator("coordinator/config.yaml")
    instance_id = coord.start("dispute_resolution", "card_dispute", case_input)
    result = coord.wait(instance_id)
"""

from coordinator.types import (
    InstanceState,
    InstanceStatus,
    WorkOrder,
    WorkOrderStatus,
    WorkOrderResult,
    Suspension,
    GovernanceTier,
    DelegationPolicy,
    Contract,
)
from coordinator.tasks import (
    TaskQueue,
    InMemoryTaskQueue,
    SQLiteTaskQueue,
    Task,
    TaskType,
    TaskStatus,
    TaskCallback,
    TaskResolution,
)
from coordinator.runtime import Coordinator

__all__ = [
    "Coordinator",
    "InstanceState",
    "InstanceStatus",
    "WorkOrder",
    "WorkOrderStatus",
    "WorkOrderResult",
    "Suspension",
    "GovernanceTier",
    "DelegationPolicy",
    "Contract",
    "TaskQueue",
    "InMemoryTaskQueue",
    "SQLiteTaskQueue",
    "Task",
    "TaskType",
    "TaskStatus",
    "TaskCallback",
    "TaskResolution",
]
