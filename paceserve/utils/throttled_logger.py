import time
from threading import Lock

from paceserve.logger import init_logger

logger = init_logger(__name__)


class ThrottledLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, interval_seconds: float=10):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, interval_seconds: float=10):
        if self._initialized:
            return
        self.interval = interval_seconds
        self.last_logged = {}
        self._initialized = True

    def log(self, key: str, message: str, log_func=logger.info, immediate=False):
        now = time.time()
        if immediate or \
           key not in self.last_logged or \
           now - self.last_logged[key] > self.interval:
                self.last_logged[key] = now
                log_func(message)
