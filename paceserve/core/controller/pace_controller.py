import ray
import queue
import time
from threading import Thread
import logging
from typing import List

from paceserve.core.controller.base_controller import BaseController
from paceserve.benchmark.config import BenchmarkConfig
from paceserve.types import InstanceType
from paceserve.disutil.instance_ref import InstanceRef
from paceserve.utils.signal_actor import SignalActor
from paceserve.utils.throttled_logger import ThrottledLogger

logger = logging.getLogger(__name__)
t_logger = ThrottledLogger()


@ray.remote
class PaceController(BaseController):
    def __init__(self, config: BenchmarkConfig, instances: List[InstanceRef], controller_stop_signal: SignalActor, instance_stop_signal: SignalActor):
        self.config = config
        self.low_idx = 0
        self.high_idx = 0
        self.transferred = 0
        
        self.high_priority_queue = queue.Queue()
        super().__init__(config, instances, controller_stop_signal, instance_stop_signal)

        self.hp_req_reschedule = Thread(target=self._get_high_priority_request, daemon=True)
        self.hp_req_reschedule.start()

    def classify(self, instances):
        self.hp_instances = []
        self.lp_instances = []
        for inst in instances:
            if inst._type == InstanceType.HIGH_PRIORITY:
                self.hp_instances.append(inst.ref)
            else:
                self.lp_instances.append(inst.ref)

    def _get_high_priority_request(self):
        waiting_ref = []
        while not self.shut_down:
            if not waiting_ref:
                for each_instance in self.lp_instances:
                    ref = each_instance._get_hp_request.remote()
                    waiting_ref.append(ref)  

            ready, waiting_ref = ray.wait(waiting_ref, num_returns=1, timeout=0.1)

            for _int_seqs in ready:
                result = ray.get(_int_seqs)
                if not result:
                    continue

                for seq in result:
                    self.high_priority_queue.put(seq)
                    self.transferred += 1
                    t_logger.log(
                        key='_get_high_priority_request',
                        message=f'controller send: {self.transferred}'
                    )

    def _schedule(self):
        if not self._request_to_schedule and self.high_priority_queue.qsize() == 0:
            time.sleep(0.001)
            return

        while self._request_to_schedule or self.high_priority_queue.qsize() > 0:
            target_hp_ref = self.hp_instances[self.high_idx % len(self.hp_instances)]
            get_ticket = ray.get(target_hp_ref._seq_polling_ticket.remote())
                
            if self.high_priority_queue.qsize() > 0:
                seq = self.high_priority_queue.get()
                target_hp_ref._add_sequence.remote(seq)
                self.high_idx += 1
            elif get_ticket:
                request = self._request_to_schedule.pop(0)
                target_hp_ref._add_request.remote(request)
                self.high_idx += 1
            else:
                if self._request_to_schedule:
                    request = self._request_to_schedule.pop(0)
                    target_lp = self.lp_instances[self.low_idx % len(self.lp_instances)]
                    self.low_idx += 1
                    target_lp._add_request.remote(request)
