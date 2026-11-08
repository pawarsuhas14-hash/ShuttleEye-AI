"""Small deterministic lock state for a manually identified baseline."""


class BaselineLock:
    def __init__(self, stable_frames=3, loss_limit=8):
        self.stable_frames = max(2, int(stable_frames))
        self.loss_limit = max(2, int(loss_limit))
        self.configured = False
        self.locked = False
        self.state = "BASELINE_SEARCH"
        self.status = "not_configured"
        self.stable_count = 0
        self.missed_frames = 0

    def configure(self):
        self.configured = True
        self.locked = False
        self.state = "BASELINE_CONFIRMATION"
        self.status = "confirming_user_marked_line"
        self.stable_count = 0
        self.missed_frames = 0

    def restore_locked(self):
        """Retain prior client-confirmed calibration across a transport reconnect."""
        self.configured = True
        self.locked = True
        self.state = "BASELINE_LOCKED"
        self.status = "retained_waiting_for_frame_revalidation"
        self.stable_count = self.stable_frames
        self.missed_frames = 0

    def observe(self, valid, reason=""):
        if not self.configured:
            return self.snapshot()
        if valid:
            self.missed_frames = 0
            if not self.locked:
                self.stable_count += 1
                if self.stable_count >= self.stable_frames:
                    self.locked = True
                    self.state = "BASELINE_LOCKED"
                    self.status = "locked"
                else:
                    self.state = "BASELINE_REACQUISITION" if self.state == "BASELINE_REACQUISITION" else "BASELINE_CONFIRMATION"
                    self.status = "confirming_user_marked_line"
            else:
                self.state = "BASELINE_LOCKED"
                self.status = "locked"
        else:
            self.stable_count = 0
            if self.locked:
                self.missed_frames += 1
                if self.missed_frames >= self.loss_limit:
                    self.locked = False
                    self.state = "BASELINE_REACQUISITION"
                    self.status = "baseline_lost_reacquisition_required"
                else:
                    self.state = "BASELINE_LOCKED"
                    self.status = f"temporarily_unverified:{reason or 'line_not_observed'}"
            else:
                self.state = "BASELINE_REACQUISITION" if self.state == "BASELINE_REACQUISITION" else "BASELINE_CONFIRMATION"
                self.status = reason or "marked_baseline_not_visible"
        return self.snapshot()

    def snapshot(self):
        return {"configured": self.configured, "locked": self.locked, "state": self.state,
                "status": self.status, "stable_count": self.stable_count,
                "missed_frames": self.missed_frames}
