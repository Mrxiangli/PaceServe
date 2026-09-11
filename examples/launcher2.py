import ray
import time



@ray.remote
class WorkerRunner:
    def __init__(self, id):
        self.id = id
        self.requests = []
        self.ref = None
        self.processed = 0
        self.start_time = None
        self.finish = False
        
    
    def set_ref(self, self_ref):
        self.ref = self_ref
    
    def is_finished(self):
        return self.finish
   
    
    def print_result(self):
        print(f"processed: {self.processed}", flush=True)
        return self.processed
        

    def _add_request(self, request):
        print(f"added 1")
        self.requests.append(request)

    def get_requests(self):
        # Return the current number of requests for debugging purposes
        return len(self.requests)

    def process_requests(self):
        # Process requests in the queue
            #pass
        if self.start_time is None:
            self.start_time = time.monotonic()
    
        elapsed_time = time.monotonic() - self.start_time
        if self.requests:
            print(f"Worker {self.id} processing {self.requests[0]} request(s)")
            self.requests.pop()
            self.processed +=1
        else:
            #print(f"Worker {self.id} has no requests to process")
            pass
        if elapsed_time < 10 :
            self.ref.process_requests.remote()
        else:
            self.finish = True


class DummyController:
    def __init__(self, workers):
        self.workers = workers
        self.generated_requests = [1, 2, 3, 4]

    def schedule_requests(self):
        total_workers = len(self.workers)
        for idx, request in enumerate(self.generated_requests):
            # Send request to the appropriate worker using round-robin scheduling
            worker_id = idx % total_workers
            ray.get(self.workers[worker_id]._add_request.remote(request))
            print(f"Added request {request} to worker {worker_id}")
            # # Fetch and log the number of requests in the worker for debugging
            # requests_num = ray.get(self.workers[worker_id].get_requests.remote())
            # print(f"Worker {worker_id} now has {requests_num} request(s)")


class WorkerLauncher:
    def __init__(self):
        ray.init(ignore_reinit_error=True)
        self.runners = self._create_runners()

    def _create_runners(self):
        # Create WorkerRunner actors
        return [WorkerRunner.remote(i) for i in range(1)]

    def run(self):
        # Start a periodic task to process requests

        for runner in self.runners:
            runner.set_ref.remote(runner)
            runner.process_requests.remote()
            #runner.print_result.remote()
            
        while True:
            if ray.get(self.runners[0].is_finished.remote()):
                processed = ray.get(self.runners[0].print_result.remote())
                break
            time.sleep(1)
            
                #time.sleep(1)  # Adjust the processing interval as needed


# Main Program
if __name__ == "__main__":
    # Step 1: Initialize the worker launcher and start workers
    launcher = WorkerLauncher()

    # Start the worker processing in a separate thread
    #import threading
    #threading.Thread(target=launcher.run, daemon=True).start()

    # Step 2: Create the controller and assign the workers
    controller = DummyController(launcher.runners)

    # Step 3: Schedule requests using the controller
    controller.schedule_requests()

    launcher.run()
    # Step 4: Keep the program running to observe outputs
    # while True:
    #     time.sleep(1)

    # Optional: Shutdown Ray
    ray.shutdown()
