import time
import pynvml
import threading
import csv

class GPUUtilizationLogger(threading.Thread):
    def __init__(self, log_file, interval_ms=10):
        """
        Initializes the GPU logger thread.
        
        :param log_file: Path to the CSV file for logging.
        :param interval_ms: Logging interval in milliseconds (default 10ms).
        """
        super().__init__()
        self.log_file = log_file
        self.interval_s = interval_ms / 1000.0  # Convert ms to seconds
        self.running = False

    def run(self):
        """Starts GPU utilization logging in a separate thread."""
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Device", "GPU_Utilization (%)"])  # CSV Header

            self.running = True
            next_time = time.monotonic()

            while self.running:
                timestamp = time.time()  # High-precision timestamp

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    writer.writerow([f"{timestamp:.6f}", i, utilization.gpu])

                # Ensure precise timing
                next_time += self.interval_s
                while time.monotonic() < next_time:
                    pass  # Busy-wait to maintain accurate intervals

        pynvml.nvmlShutdown()
        print(f"GPU logging stopped. Data saved to {self.log_file}")

    def stop(self):
        """Stops the logging thread gracefully."""
        self.running = False

# Example usage
if __name__ == "__main__":
    logger = GPUUtilizationLogger(f"gpu_log_{time.time()}.csv", interval_ms=20)
    logger.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread running
    except KeyboardInterrupt:
        print("Stopping GPU logger...")
        logger.stop()
        logger.join()
