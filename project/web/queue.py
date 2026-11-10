"""One durable worker; process locks prevent accidental multi-worker inference."""
from __future__ import annotations

import fcntl
from pathlib import Path
import threading

from .storage import ApiError


class ProcessLock:
    def __init__(self, path: Path):
        self.path, self.handle = path, None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.handle = self.path.open("a")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Web API 已运行；仅允许一个进程和一个worker。") from exc

    def release(self):
        if self.handle:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class TaskRunner:
    def __init__(self, store, engine_factory=None, *, cleanup_interval=60):
        self.store, self.engine_factory = store, engine_factory
        self.engine = None
        self.cleanup_interval = cleanup_interval
        self.stopping, self.wake = threading.Event(), threading.Event()
        self.worker = self.cleaner = None

    def _engine(self):
        if self.engine is None:
            if self.engine_factory:
                self.engine = self.engine_factory()
            else:
                from project.core.web_workflow import WebPlanningEngine
                self.engine = WebPlanningEngine()
        return self.engine

    def process_once(self):
        task = self.store.claim_task()
        if task is None:
            return False
        try:
            result = self._engine().process(task["kind"], task["state"], task["payload"], task["files"])
            if not isinstance(result, dict):
                raise ValueError("Engine result must be a dictionary")
        except Exception as exc:
            # Model/load errors can contain raw text, tokens or private paths.
            self.store.finish_task(task["id"], result=getattr(exc, "partial_state", None), error={"code": getattr(exc, "code", "model_task_failed"), "message": getattr(exc, "public_message", "任务未完成，已保留此前状态；未自动重试。"), "type": type(exc).__name__})
        else:
            self.store.finish_task(task["id"], result=result)
        return True

    def start(self):
        self.store.interrupt_running()
        self.store.cleanup()
        self.worker = threading.Thread(target=self._work, name="career-model-worker", daemon=True)
        self.cleaner = threading.Thread(target=self._clean, name="career-retention", daemon=True)
        self.worker.start()
        self.cleaner.start()

    def _work(self):
        while not self.stopping.is_set():
            try:
                if self.process_once():
                    continue
            except ApiError:
                # Leave queued state durable while storage is unavailable.
                pass
            self.wake.wait(1)
            self.wake.clear()

    def _clean(self):
        while not self.stopping.wait(self.cleanup_interval):
            try:
                self.store.cleanup()
            except ApiError:
                # Failures remain visible to API checks and next cleanup retries.
                pass

    def stop(self):
        self.stopping.set()
        self.wake.set()
        for thread in (self.worker, self.cleaner):
            if thread:
                thread.join()
