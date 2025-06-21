import threading
from typing import Dict, Tuple

class TaskManager:
    """Simple task manager to track threads with cancellation events."""
    def __init__(self):
        self._tasks: Dict[str, Tuple[threading.Thread, threading.Event]] = {}
        self._lock = threading.Lock()

    def register(self, task_id: str, thread: threading.Thread, cancel_event: threading.Event) -> None:
        with self._lock:
            self._tasks[task_id] = (thread, cancel_event)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            entry = self._tasks.get(task_id)
            if not entry:
                return False
            thread, event = entry
            event.set()
        thread.join(timeout=1)
        with self._lock:
            self._tasks.pop(task_id, None)
        return True

    def cancel_all(self) -> None:
        with self._lock:
            task_ids = list(self._tasks.keys())
        for task_id in task_ids:
            self.cancel(task_id)

    def get(self, task_id: str):
        with self._lock:
            return self._tasks.get(task_id)
