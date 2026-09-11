from typing import List
from paceserve.core.controller.base_controller import BaseController
from paceserve.benchmark.config import BenchmarkConfig
import ray
from paceserve.disutil.instance_ref import InstanceRef
from paceserve.utils.signal_actor import SignalActor

@ray.remote
class RoundRobinController(BaseController):
    def __init__(self, config: BenchmarkConfig, instances: List[InstanceRef], controller_stop_signal: SignalActor, instance_stop_signal: SignalActor):
        super().__init__(config, instances, controller_stop_signal, instance_stop_signal)
        self.config = config
    
    def classify(self, instances):
        pass
    
    def _schedule(self):
        while self._request_to_schedule:
            total_instance = len(self.instances)
            request = self._request_to_schedule.pop(0)
            self.instances[self.idx % total_instance]._add_request.remote(request) 
            self.idx += 1
