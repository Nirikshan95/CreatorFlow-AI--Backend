from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading
import uuid
from contextvars import ContextVar

class WorkflowLogger:
    """Single-file workflow logger with per-generation in-memory buffers."""
    
    def __init__(self, log_path: str = "data/generation.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffers: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        self._current_generation_id: ContextVar[Optional[str]] = ContextVar(
            "workflow_generation_id",
            default=None
        )

    def _resolve_generation_id(self, generation_id: Optional[str] = None) -> str:
        resolved = generation_id or self._current_generation_id.get()
        if not resolved:
            resolved = "global"
        return resolved

    def _path_for_generation(self, generation_id: str) -> Path:
        # Always use one shared log file on disk.
        return self.log_path

    def set_current_generation(self, generation_id: str):
        self._current_generation_id.set(generation_id)

    def start_generation(self, generation_id: Optional[str] = None) -> str:
        generation_id = generation_id or str(uuid.uuid4())
        self.set_current_generation(generation_id)
        msg = f"=== New Generation Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
        with self._lock:
            # New run: clear previous in-memory buffers and start fresh.
            self.buffers.clear()
            self.buffers[generation_id] = [msg]
        log_file = self._path_for_generation(generation_id)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"{msg}\n\n")
        return generation_id

    def reset_log(self, generation_id: Optional[str] = None):
        """Backward-compatible alias used by existing workflow code."""
        self.start_generation(generation_id)

    def log_step(self, step_name: str, status: str, details: str = "", generation_id: Optional[str] = None):
        """Logs a specific step status."""
        generation_id = self._resolve_generation_id(generation_id)
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [{step_name.upper()}] [{status.upper()}]"
        message = f"{prefix} {details}".strip()

        with self._lock:
            self.buffers.setdefault(generation_id, []).append(message)

        log_file = self._path_for_generation(generation_id)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
            if status.lower() == "error":
                f.write("-" * 20 + "\n")

    def get_new_messages(self, generation_id: Optional[str] = None) -> List[str]:
        """Returns and clears buffered messages for one generation."""
        generation_id = self._resolve_generation_id(generation_id)
        with self._lock:
            msgs = list(self.buffers.get(generation_id, []))
            self.buffers[generation_id] = []
        return msgs

    def read_log(self, generation_id: Optional[str] = None) -> str:
        """Reads current single log file content."""
        log_file = self.log_path
        if not log_file.exists():
            return "No log file found."
        return log_file.read_text(encoding="utf-8")

    def end_generation(self, generation_id: Optional[str] = None):
        """Release in-memory buffer for a generation after request completion."""
        generation_id = self._resolve_generation_id(generation_id)
        with self._lock:
            self.buffers.pop(generation_id, None)

workflow_logger = WorkflowLogger()
