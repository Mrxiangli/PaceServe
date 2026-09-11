from abc import ABC, abstractmethod
import time
from typing import List
from threading import Thread

import ray
from paceserve.benchmark.entities import Request
from paceserve.benchmark.config import BenchmarkConfig
from paceserve.benchmark.utils.random import set_seeds
from paceserve.benchmark.request_generator import RequestGeneratorRegistry
from paceserve.logger import init_logger
from paceserve.utils import env
from paceserve.disutil.instance_ref import InstanceRef
from paceserve.utils.signal_actor import SignalActor
from paceserve.utils.throttled_logger import ThrottledLogger

logger = init_logger(__name__)
t_logger = ThrottledLogger()


class BaseController(ABC):
    def __init__(self, config: BenchmarkConfig, instances: List[InstanceRef], controller_stop_signal: SignalActor, instance_stop_signal: SignalActor):
        self.config = config
        self.type_assigned = False
        self.shut_down = False
        self.idx = 0
        self.start_time = None
        self.generated_requests = []
        self.total_request_generated = 0
        self._request_to_schedule = []
        self._classify_instances(instances)
        self.instance_stop_signal = instance_stop_signal
        self.controller_stop_signal = controller_stop_signal
        self.completed_request_collect = Thread(target=self._get_completed_request, daemon=True)
        self.controller_stop = Thread(target=self._controller_stop, daemon=True)
        self.controller_stop.start()
        self.in_window_request_ids = set()
        self.warmup_time = self.config.warmup_time
        self.window_duration = self.config.warmup_time + self.config.time_limit
        self.static_profile = self.config.static_profile

        if not env.Options.SHOW_DEBUG_INFO:
            i_types = {i.id: i._type for i in instances}
            logger.debug(f"instance info: {i_types}")
    
    @abstractmethod
    def classify(self, instances):
        pass

    def _classify_instances(self, instances):
        self.instances = [inst.ref for inst in instances]
        self.classify(instances)
    
    def _controller_status(self):
        return self.shut_down
    
    def _shut_down(self):
        self.shut_down = True
    
    def _controller_stop(self):
        ray.get(self.controller_stop_signal.wait.remote())
        self._shut_down()

    def _get_completed_request(self):
        total_finished = 0
        total_finished_in_window = 0
        while not self.shut_down and total_finished < self.total_request_generated:
            total_finished = 0
            refs = []
            for instance in self.instances:
                refs.append(instance._sequence_finished_and_aborted.remote())
            for result in ray.get(refs):
                total_finished += result['total_count']
                common_ids = set(self.in_window_request_ids) & set(result['ids'])
                count = len(common_ids)
                total_finished_in_window += count
                self.in_window_request_ids -= result['ids']

            t_logger.log(
                key='_get_completed_request',
                message=f'{total_finished=} | {total_finished_in_window=} | remaining requests in the window = {len(self.in_window_request_ids)}'
            )
            if len(self.in_window_request_ids) == 0:
                logger.info('All the requests in the window finished, shutting down...')
                break
        self.instance_stop_signal.send.remote()
        self._shut_down()

    def get_output_ids(self):
        refs = []
        for instance in self.instances:
            refs.append(instance._get_inference_result.remote())
        return ray.get(refs)

    def generate_request(self):
        set_seeds(self.config.seed)
        request_generator = RequestGeneratorRegistry.get(
            self.config.request_generator_config.get_type(),
            self.config.request_generator_config,
        )
        requests: List[Request] = request_generator.generate()
        self.generated_requests = sorted(
            requests,
            key=lambda req: req.arrived_at,
            reverse=True
        )
        self.total_request_generated = len(self.generated_requests)

        for r in self.generated_requests:
            if (self.warmup_time <= r.arrived_at <= self.window_duration) or self.config.static_profile: 
                r._in_window = True
                self.in_window_request_ids |= {r.id}
        logger.info(
            "Controller generated %d requests | In-window requests %d",
            self.total_request_generated,
            len(self.in_window_request_ids)
        )
        assert len(self.in_window_request_ids) > 1, \
            f'There should be enough requests within the measurement window!'

    def _type_assignment_finished(self):
        return self.type_assigned

    @abstractmethod
    def _schedule(self):
        pass

    def schedule_requests(self):
        if not self.start_time:
            self.start_time = time.monotonic()

        self.completed_request_collect.start()
        while not self.shut_down:
            # Get the latest requests out and put them in the queue
            now = time.monotonic()
            if self.generated_requests:
                req = self.generated_requests[-1]
                if self.start_time + req.arrived_at < now:
                    req = self.generated_requests.pop()
                    self._request_to_schedule.append(req)

            self._schedule()
