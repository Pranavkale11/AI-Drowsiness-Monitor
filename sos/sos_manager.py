"""
sos/sos_manager.py — SOS State Machine
========================================
A pure-Python, thread-safe state machine that controls the SOS alert
lifecycle.  Has NO dependency on Streamlit — all Streamlit integration lives
in app.py, which stores the SosManager instance via ``@st.cache_resource``.

States
------
    IDLE        : Normal monitoring, no alert pending.
    COUNTDOWN   : Severe drowsiness detected; countdown running.
                  Driver or the system can cancel before it reaches 0.
    TRIGGERED   : Countdown expired; SMS has been (or is being) sent.
    COOLDOWN    : Post-trigger cooldown period; no new auto SOS will fire.

State Transitions
-----------------
    IDLE ──(severe drowsiness detected)──► COUNTDOWN
    COUNTDOWN ──(driver recovers / manual cancel)──► IDLE
    COUNTDOWN ──(countdown expires)──► TRIGGERED
    TRIGGERED ──(enter_cooldown)──► COOLDOWN
    COOLDOWN ──(cooldown expires + drowsiness)──► IDLE  (then restarts)
    COOLDOWN ──(driver recovers + cooldown expired)──► IDLE

Thread Safety
-------------
    All public methods acquire ``self._lock`` so that detection threads and
    the Streamlit rerun thread can safely call them concurrently.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple


class SosState:
    """String constants for SOS state names."""
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    TRIGGERED = "TRIGGERED"
    COOLDOWN = "COOLDOWN"


class SosManager:
    """Thread-safe SOS alert state machine.

    Typical usage inside Streamlit (via @st.cache_resource):

        @st.cache_resource(show_spinner=False)
        def _get_sos_manager() -> SosManager:
            return SosManager()

        sm = _get_sos_manager()
        action = sm.notify_severe_drowsiness()
        if action == "TRIGGER_SOS":
            send_sms(...)
            sm.enter_cooldown()
    """

    # Class-level defaults (can be overridden per instance)
    DEFAULT_COUNTDOWN_SECONDS: int = 0
    DEFAULT_COOLDOWN_SECONDS: int = 300  # 5 minutes

    def __init__(
        self,
        countdown_seconds: int = DEFAULT_COUNTDOWN_SECONDS,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._lock = threading.Lock()
        self._state: str = SosState.IDLE
        self._countdown_start: Optional[float] = None
        self._cooldown_until: Optional[float] = None
        self._countdown_seconds: int = max(0, min(120, countdown_seconds))
        self._cooldown_seconds: int = max(60, cooldown_seconds)
        self._trigger_count: int = 0  # lifetime count of SOS triggers

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state string (thread-safe read)."""
        with self._lock:
            return self._state

    @property
    def countdown_seconds(self) -> int:
        """Configured countdown duration in seconds."""
        with self._lock:
            return self._countdown_seconds

    @countdown_seconds.setter
    def countdown_seconds(self, value: int) -> None:
        with self._lock:
            self._countdown_seconds = max(0, min(120, int(value)))

    @property
    def trigger_count(self) -> int:
        """Total number of SOS triggers this session."""
        with self._lock:
            return self._trigger_count

    # ------------------------------------------------------------------
    # Core state transition methods
    # ------------------------------------------------------------------

    def notify_severe_drowsiness(self, now: Optional[float] = None) -> Optional[str]:
        """Notify the manager that severe/sustained drowsiness is active.

        Call this on every frame where ``closed_counter >= SOS_FRAMES``.

        Returns:
            "COUNTDOWN_STARTED" — countdown just began (IDLE → COUNTDOWN).
            "TRIGGER_SOS"       — countdown expired; send the SMS now.
            None                — no new action required.
        """
        if now is None:
            now = time.time()

        with self._lock:
            if self._state == SosState.IDLE:
                if self._countdown_seconds <= 0:
                    self._state = SosState.TRIGGERED
                    self._trigger_count += 1
                    return "TRIGGER_SOS"
                self._state = SosState.COUNTDOWN
                self._countdown_start = now
                return "COUNTDOWN_STARTED"

            if self._state == SosState.COUNTDOWN:
                if self._countdown_start is None:
                    # Safety reset
                    self._countdown_start = now
                    return None
                elapsed = now - self._countdown_start
                if elapsed >= self._countdown_seconds:
                    self._state = SosState.TRIGGERED
                    self._trigger_count += 1
                    return "TRIGGER_SOS"
                return None  # still counting down

            if self._state == SosState.COOLDOWN:
                # Check if cooldown has expired naturally
                if self._cooldown_until is not None and now >= self._cooldown_until:
                    # Cooldown over — restart the process
                    self._state = SosState.IDLE
                    self._cooldown_until = None
                return None

            # TRIGGERED state — SMS is being sent; wait for enter_cooldown()
            return None

    def notify_driver_active(self, now: Optional[float] = None) -> Optional[str]:
        """Notify the manager that the driver is awake (eyes open).

        Call this whenever ``closed_counter`` resets to 0.

        Returns:
            "COUNTDOWN_CANCELLED" — driver recovered during countdown.
            "COOLDOWN_ENDED"      — cooldown just expired naturally.
            None                  — no state change.
        """
        if now is None:
            now = time.time()

        with self._lock:
            if self._state == SosState.COUNTDOWN:
                self._state = SosState.IDLE
                self._countdown_start = None
                return "COUNTDOWN_CANCELLED"

            if self._state == SosState.COOLDOWN:
                if self._cooldown_until is not None and now >= self._cooldown_until:
                    self._state = SosState.IDLE
                    self._cooldown_until = None
                    return "COOLDOWN_ENDED"

            return None

    def cancel_countdown(self) -> bool:
        """Manually cancel the active SOS countdown (driver presses Cancel).

        Returns:
            True  — countdown was active and has been cancelled.
            False — no countdown was running.
        """
        with self._lock:
            if self._state == SosState.COUNTDOWN:
                self._state = SosState.IDLE
                self._countdown_start = None
                return True
            return False

    def enter_cooldown(self, now: Optional[float] = None) -> None:
        """Move from TRIGGERED → COOLDOWN after the SMS has been sent.

        Call this immediately after a successful (or failed) SMS dispatch to
        prevent repeated messages.
        """
        if now is None:
            now = time.time()
        with self._lock:
            self._state = SosState.COOLDOWN
            self._cooldown_until = now + self._cooldown_seconds
            self._countdown_start = None

    def trigger_manual(
        self, now: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Attempt to trigger a manual SOS.

        Returns:
            (True, "TRIGGER_MANUAL_SOS")  — caller should send the SMS.
            (False, reason_str)           — blocked (e.g. cooldown active).
        """
        if now is None:
            now = time.time()
        with self._lock:
            if self._state == SosState.COOLDOWN:
                remaining = (
                    max(0.0, self._cooldown_until - now)
                    if self._cooldown_until
                    else 0.0
                )
                return (
                    False,
                    f"SOS is in cooldown. Please wait {int(remaining)}s before sending another alert.",
                )
            # Allow from IDLE, COUNTDOWN, or even TRIGGERED
            self._state = SosState.TRIGGERED
            self._trigger_count += 1
            return True, "TRIGGER_MANUAL_SOS"

    def reset(self) -> None:
        """Force-reset to IDLE (used on app restart or for testing)."""
        with self._lock:
            self._state = SosState.IDLE
            self._countdown_start = None
            self._cooldown_until = None

    # ------------------------------------------------------------------
    # Query helpers (safe to call at any time)
    # ------------------------------------------------------------------

    def get_countdown_remaining(self, now: Optional[float] = None) -> float:
        """Return seconds left in the current countdown (0.0 if not active)."""
        if now is None:
            now = time.time()
        with self._lock:
            if self._state != SosState.COUNTDOWN or self._countdown_start is None:
                return 0.0
            return max(0.0, self._countdown_seconds - (now - self._countdown_start))

    def get_cooldown_remaining(self, now: Optional[float] = None) -> float:
        """Return seconds left in the cooldown period (0.0 if not in cooldown)."""
        if now is None:
            now = time.time()
        with self._lock:
            if self._state != SosState.COOLDOWN or self._cooldown_until is None:
                return 0.0
            return max(0.0, self._cooldown_until - now)

    def is_in_countdown(self) -> bool:
        """Return True when a countdown is actively running."""
        return self.state == SosState.COUNTDOWN

    def is_in_cooldown(self) -> bool:
        """Return True when the post-SOS cooldown is active."""
        return self.state == SosState.COOLDOWN

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SosManager(state={self._state!r}, "
            f"countdown_s={self._countdown_seconds}, "
            f"cooldown_s={self._cooldown_seconds}, "
            f"triggers={self._trigger_count})"
        )
