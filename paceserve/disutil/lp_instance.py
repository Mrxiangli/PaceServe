import logging

import ray

from paceserve.disutil.instance import Instance
from paceserve.config.config import LowPrioritySchedulerConfig
from paceserve.utils.throttled_logger import ThrottledLogger

logger = logging.getLogger(__name__)
t_logger = ThrottledLogger()


@ray.remote
class LPInstance(Instance):

    def __init__(
        self,
        *args, **kwargs) -> None:
        args_list = list(args)  
        config = args_list[2]
        config.scheduler_config = LowPrioritySchedulerConfig(
            max_batched_tokens=config.scheduler_config.max_batched_tokens,
            max_num_seqs=config.scheduler_config.max_num_seqs,
            val_func=config.scheduler_config.val_func,
            allow_drop_requests = config.scheduler_config.allow_drop_requests,
            request_drop_threshold = config.scheduler_config.request_drop_threshold,
            ttft_slo = config.scheduler_config.ttft_slo,
            tbt_slo = config.scheduler_config.tbt_slo
        )
        args_list[2] = config

        super().__init__(*args_list, **kwargs)
        self.transferred = 0

    def _get_hp_request(self):
        result = []
        while self.llm_engine.hp_req_queue:
            seq = self.llm_engine.hp_req_queue.pop(0)
            result.append(seq)
            self.transferred += 1
            t_logger.log(
                key='_get_hp_request',
                message=f'lp transferred total: {self.transferred}'
            )
        return result
