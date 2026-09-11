from threading import Thread
import ray
import queue
from typing import List
import time

from paceserve.core.controller.base_controller import BaseController
from paceserve.benchmark.config import BenchmarkConfig
from paceserve.types import InstanceType
from paceserve.logger import init_logger
from paceserve.disutil.instance_ref import InstanceRef
from paceserve.utils.signal_actor import SignalActor

logger = init_logger(__name__)


@ray.remote(max_concurrency=10)
class DistServeController(BaseController):
    def __init__(self, config: BenchmarkConfig, instances: List[InstanceRef], controller_stop_signal: SignalActor, instance_stop_signal: SignalActor):
        super().__init__(config, instances, controller_stop_signal, instance_stop_signal)

        self.prefill_idx = 0
        self.instance_broadcast()

        logger.debug(f'Total prefill instances = {len(self.prefill_instances)} | decode instances = {len(self.decode_instances)}')
        
    def classify(self, instances):
        self.prefill_instances = []
        self.decode_instances = []
        for inst in instances:
            if inst._type == InstanceType.PREFILL:
                self.prefill_instances.append(inst.ref)
            else:
                self.decode_instances.append(inst.ref)
    
    def instance_broadcast(self):
        pending_refs = []
        for prefill_instance in self.prefill_instances:
            ref = prefill_instance.add_decode_instances.remote(self.decode_instances)      
            pending_refs.append(ref)
        # ensure that decode instances are broadcasted to prefill instances before further process  
        ray.get(pending_refs)
  
    def _schedule(self):
        # start metadata retrival thread to collect metadata of request that finished their prefill
        while self._request_to_schedule:
            request = self._request_to_schedule.pop(0)
            self.prefill_instances[self.prefill_idx % len(self.prefill_instances)]._add_request.remote(request) 
            self.prefill_idx += 1

