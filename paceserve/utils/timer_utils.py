import time
from paceserve.logger import init_logger

logger = init_logger(__name__)

class Timer:
    def __init__(self, name="Timer"):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self  # Allows access to execution time later if needed
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.execution_time = self.end_time - self.start_time
        logger.info(f"{self.name} execution time: {self.execution_time:.6f} seconds")
