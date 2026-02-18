"""
Cognitive Core — Logic Circuit Breakers (S-010)

Monitors per-domain/primitive quality and auto-upgrades governance tier
when quality degrades. Uses a sliding window (not consecutive count)
for graduated response.

Window behavior:
  - Tracks last N results (default 20) per (domain, primitive) pair
  - Each result is 1 (low confidence) or 0 (normal)
  - If low-confidence rate > 50%: upgrade to spot_check
  - If low-confidence rate > 80%: upgrade to gate
  - Auto-recovers when rate drops below thresholds
  - Persistence via SQLite for survival across restarts

Usage:
    from engine.logic_breaker import LogicCircuitBreaker, get_logic_breaker

    breaker = LogicCircuitBreaker()
    breaker.record("card_dispute", "classify", confidence=0.3, floor=0.5)

    override = breaker.get_tier_override("card_dispute")
    # Returns "spot_check", "gate", or None (no override)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cognitive_core.logic_breaker")


@dataclass
class BreakerState:
    """State for a single (domain, primitive) circuit breaker."""
    domain: str
    primitive: str
    window: deque  # deque of (timestamp, is_low_confidence: bool)
    window_size: int
    trip_count: int = 0       # Total number of times this breaker tripped
    last_trip_at: float = 0.0
    override_tier: str = ""   # Current tier override ("" = no override)

    @property
    def low_confidence_rate(self) -> float:
        if not self.window:
            return 0.0
        return sum(1 for _, low in self.window if low) / len(self.window)

    @property
    def window_fill(self) -> int:
        return len(self.window)


class LogicCircuitBreaker:
    """
    Sliding-window circuit breaker for domain/primitive quality.

    Monitors confidence levels and auto-upgrades governance tier
    when a domain's primitives produce too many low-confidence results.
    """

    def __init__(
        self,
        window_size: int = 20,
        spot_check_threshold: float = 0.50,
        gate_threshold: float = 0.80,
        confidence_floor: float = 0.5,
        min_samples: int = 5,
        db_path: str | None = None,
    ):
        self.window_size = window_size
        self.spot_check_threshold = spot_check_threshold
        self.gate_threshold = gate_threshold
        self.default_confidence_floor = confidence_floor
        self.min_samples = min_samples
        self.db_path = db_path

        self._breakers: dict[tuple[str, str], BreakerState] = {}
        self._lock = threading.Lock()

        # Load persisted state if DB provided
        if self.db_path:
            self._init_db()
            self._load_state()

    def record(
        self,
        domain: str,
        primitive: str,
        confidence: float,
        floor: float | None = None,
    ) -> str | None:
        """
        Record a result. Returns new tier override if state changed, else None.

        Args:
            domain: Domain name
            primitive: Primitive name (classify, investigate, etc.)
            confidence: The confidence score from the LLM output
            floor: Confidence floor (below this = low confidence).
                   Defaults to self.default_confidence_floor.
        """
        floor = floor if floor is not None else self.default_confidence_floor
        is_low = confidence < floor

        with self._lock:
            key = (domain, primitive)
            state = self._breakers.get(key)
            if state is None:
                state = BreakerState(
                    domain=domain,
                    primitive=primitive,
                    window=deque(maxlen=self.window_size),
                    window_size=self.window_size,
                )
                self._breakers[key] = state

            # Add to sliding window
            state.window.append((time.time(), is_low))

            # Evaluate thresholds (only if enough samples)
            rate = state.low_confidence_rate
            old_override = state.override_tier

            if state.window_fill >= self.min_samples:
                if rate >= self.gate_threshold:
                    state.override_tier = "gate"
                elif rate >= self.spot_check_threshold:
                    state.override_tier = "spot_check"
                else:
                    state.override_tier = ""  # Auto-recover
            # else: not enough data, keep current state

            # Track trip events
            new_override = state.override_tier
            if new_override and new_override != old_override:
                state.trip_count += 1
                state.last_trip_at = time.time()
                logger.warning(
                    "Logic breaker TRIPPED: %s/%s rate=%.1f%% → tier=%s (trip #%d)",
                    domain, primitive, rate * 100, new_override, state.trip_count,
                )

            if old_override and not new_override:
                logger.info(
                    "Logic breaker RECOVERED: %s/%s rate=%.1f%%",
                    domain, primitive, rate * 100,
                )

            # Persist
            if self.db_path:
                self._save_state(state)

            if new_override != old_override:
                return new_override or None
            return None

    def get_tier_override(self, domain: str) -> str | None:
        """
        Get the most severe tier override for a domain.
        Checks all primitives and returns the highest override.
        """
        with self._lock:
            worst = ""
            tier_rank = {"": 0, "spot_check": 1, "gate": 2}

            for (d, p), state in self._breakers.items():
                if d == domain and state.override_tier:
                    if tier_rank.get(state.override_tier, 0) > tier_rank.get(worst, 0):
                        worst = state.override_tier

            return worst if worst else None

    def get_state(self, domain: str, primitive: str) -> dict[str, Any] | None:
        """Get breaker state for inspection/metrics."""
        with self._lock:
            state = self._breakers.get((domain, primitive))
            if state is None:
                return None
            return {
                "domain": state.domain,
                "primitive": state.primitive,
                "low_confidence_rate": round(state.low_confidence_rate, 3),
                "window_fill": state.window_fill,
                "window_size": state.window_size,
                "override_tier": state.override_tier,
                "trip_count": state.trip_count,
                "last_trip_at": state.last_trip_at,
            }

    def get_all_states(self) -> list[dict[str, Any]]:
        """Get all breaker states for health/metrics endpoints."""
        with self._lock:
            return [
                {
                    "domain": s.domain,
                    "primitive": s.primitive,
                    "low_confidence_rate": round(s.low_confidence_rate, 3),
                    "window_fill": s.window_fill,
                    "override_tier": s.override_tier,
                    "trip_count": s.trip_count,
                }
                for s in self._breakers.values()
            ]

    def reset(self, domain: str, primitive: str | None = None):
        """
        Reset breaker state. If primitive is None, reset all for the domain.
        """
        with self._lock:
            keys_to_remove = []
            for (d, p) in self._breakers:
                if d == domain and (primitive is None or p == primitive):
                    keys_to_remove.append((d, p))

            for key in keys_to_remove:
                del self._breakers[key]
                if self.db_path:
                    self._delete_state(key[0], key[1])

            logger.info("Logic breaker RESET: domain=%s primitive=%s", domain, primitive or "ALL")

    def reset_all(self):
        """Reset all breaker states."""
        with self._lock:
            self._breakers.clear()
            if self.db_path:
                self._clear_all_state()
            logger.info("Logic breaker RESET ALL")

    # ── Persistence ──────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logic_breaker_state (
                domain TEXT NOT NULL,
                primitive TEXT NOT NULL,
                override_tier TEXT DEFAULT '',
                trip_count INTEGER DEFAULT 0,
                last_trip_at REAL DEFAULT 0,
                window_json TEXT DEFAULT '[]',
                updated_at REAL DEFAULT 0,
                PRIMARY KEY (domain, primitive)
            )
        """)
        conn.commit()
        conn.close()

    def _save_state(self, state: BreakerState):
        import json
        conn = sqlite3.connect(self.db_path)
        window_data = json.dumps(list(state.window))
        conn.execute("""
            INSERT OR REPLACE INTO logic_breaker_state
            (domain, primitive, override_tier, trip_count, last_trip_at, window_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            state.domain, state.primitive, state.override_tier,
            state.trip_count, state.last_trip_at, window_data, time.time(),
        ))
        conn.commit()
        conn.close()

    def _load_state(self):
        import json
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM logic_breaker_state").fetchall()
        for row in rows:
            domain, primitive, override_tier, trip_count, last_trip_at, window_json, _ = row
            window_data = json.loads(window_json) if window_json else []
            window = deque(maxlen=self.window_size)
            for item in window_data[-self.window_size:]:
                window.append(tuple(item))
            state = BreakerState(
                domain=domain,
                primitive=primitive,
                window=window,
                window_size=self.window_size,
                trip_count=trip_count,
                last_trip_at=last_trip_at,
                override_tier=override_tier,
            )
            self._breakers[(domain, primitive)] = state
        conn.close()
        if rows:
            logger.info("Loaded %d logic breaker states from DB", len(rows))

    def _delete_state(self, domain: str, primitive: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "DELETE FROM logic_breaker_state WHERE domain=? AND primitive=?",
            (domain, primitive),
        )
        conn.commit()
        conn.close()

    def _clear_all_state(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM logic_breaker_state")
        conn.commit()
        conn.close()


# ═══════════════════════════════════════════════════════════════════
# Module-level Singleton
# ═══════════════════════════════════════════════════════════════════

_instance: LogicCircuitBreaker | None = None


def get_logic_breaker(**kwargs) -> LogicCircuitBreaker:
    global _instance
    if _instance is None:
        _instance = LogicCircuitBreaker(**kwargs)
    return _instance


def reset_logic_breaker():
    global _instance
    _instance = None
