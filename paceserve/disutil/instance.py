from abc import ABC
import logging
import time
import torch
import ray
from threading import Thread
from typing import List
from rouge_score import rouge_scorer


import torch

from ray.util.metrics import Gauge

from paceserve import LLMEngine, SamplingParams
from paceserve.benchmark.config import BenchmarkConfig
from paceserve.benchmark.entities import Request
from paceserve.benchmark.entities.request import PromptRequest, DatasetRequest, QARequest
from paceserve.config import ReplicaConfig, ParallelConfig
from paceserve.config import InstanceConfig
from paceserve.utils.signal_actor import SignalActor
from paceserve.core.datatypes.request_output import RequestOutput
from paceserve.core.datatypes.request_output import PromptResponse
from paceserve.ai_utils.ai_helper import AIHelper
from paceserve.utils.throttled_logger import ThrottledLogger

logger = logging.getLogger(__name__)
t_logger = ThrottledLogger()


class Instance(ABC):

    def __init__(
        self,
        replica_id: int,
        instance_id: int,
        config: BenchmarkConfig,
        parallel_config: ParallelConfig,
        instance_config: InstanceConfig,
        ready_signal: SignalActor,
        stop_signal: SignalActor,
        use_logits: bool
    ) -> None:
        self.start_time = None
        self.replica_id = replica_id
        self.instance_id = instance_id
        self.config = config
        self.instance_type = instance_config._type
        self.calculate_rouge = instance_config.calculate_rouge
        self.ready_signal = ready_signal
        self.stop_signal = stop_signal
        self.terminate = False
        self.requests = []
        self.output_list = []
        self._new_finished_ids = set()
        self.parallel_engine = (parallel_config.world_size > 1)
        self.gpu_device_id = instance_config.resource_mapping[0][1]
        self.device = torch.device(f'cuda:{self.gpu_device_id}')
        self._running = False
        self.total_req =0
        self.use_logits = use_logits

        logger.debug(f"instance {self.instance_id} mapping: {instance_config.resource_mapping}")

        self.waitig_stop = Thread(target=self. _waiting_stop_signal, daemon=True)
        self.waitig_stop.start()
        self.run_loop_finished = False
        
        if self.calculate_rouge:
            self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.rouge_dic = {}

        # the replica config is used to set up metric store, passing instance id
        replica_config = ReplicaConfig(
            replica_id=self.instance_id, 
            output_dir=self.config.output_dir,
        )
        self.instance_config = instance_config
        
        system_config = self.config.create_instance_system_config(replica_config, instance_config, parallel_config)
        self.llm_engine = LLMEngine.from_system_config(system_config)
        
        self.gpu_max_blocks = Gauge(
            "gpu_max_blocks",
            description="Maximum number of GPU blocks available.",
        )
        self.gpu_free_blocks = Gauge(
            "gpu_free_blocks",
            description="Current number of free GPU blocks available.",
        )
        
        # Singletone class: For estimating batch runtime
        _ai_helper = AIHelper()
        _ai_helper.initialize(config.model_config, config.scheduler_config, system_config.batchregression_config)
 
        self._run_finish = SignalActor.remote()
 
    def _set_instance_type(self, instance_type):
        self.instance_type = instance_type
    
    def _get_instance_type(self):
        return self.instance_type

    def _get_instance_id(self):
        return self.instance_id

    def get_info(self):
        return (self.instance_id, self.instance_type)

    def _set_ref(self, self_ref):
        self.ref = self_ref
    
    def _add_request(self, request: Request):
        self.requests.append(request)
    
    def _is_finished(self):
        return self.terminate

    def finished_seq(self, output: RequestOutput):
        self.output_list.append(output)
        self._new_finished_ids |= {output.seq_id}

    def _get_input_params(self, request: Request, current_time: float):
        if isinstance(request, PromptRequest):
            sampling_params = SamplingParams(temperature=0.85,
                                             top_p=0.95, max_tokens=20)
            return {
                'prompt': request.prompt,
                'sampling_params': sampling_params
            }
        if isinstance(request, QARequest):
            sampling_params = SamplingParams(temperature=0.85,
                                             top_p=0.95, max_tokens=1)
            return {
                'prompt': request.text,
                'prompt_token_ids': request.prompt_ids,
                'sampling_params': sampling_params
            }
        if isinstance(request, DatasetRequest):
            sampling_params = SamplingParams(temperature=0.85,
                                             top_p=0.95, max_tokens=10)
            return {
                'prompt': request.text,
                'prompt_token_ids': request.prompt_ids,
                'sampling_params': sampling_params
            }

        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,
            temperature=0,
            top_p=1.0,
        )
        prompt_token_ids = [1] * request.num_prefill_tokens

        return {
            "prompt": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": sampling_params,
            "arrival_time": current_time,
            "seq_id": request.id,
            "is_in_window": request.is_in_window,
        }

    def warmup(self) -> None:
        dummy_Request = Request(arrived_at=time.monotonic(), num_prefill_tokens=1000, num_decode_tokens=200)
        self.llm_engine.add_request(**self._get_input_params(dummy_Request, float('inf')))

        is_completed = False
        while not is_completed:
            step_outputs = self.llm_engine.step()
            is_completed = step_outputs[0].finished
        self.llm_engine.reset_metrics()
    
    def _prepare_run(self):
        if self.config.enable_profiling:
            self.llm_engine.start_profiling()
        
        self.num_processed_requests = 0
        self.num_steps = 0
        self.num_token_processed = 0
        self.nll_sum = 0.0

    def _on_step_complete(self):
        pass

    def _process_pending_sequences(self):
        """Install transferred sequences on the engine's run thread."""
        pass

    def _run(self) -> None:
        self.start_time = time.monotonic()

        while not self.terminate:
            while self.requests:
                request = self.requests.pop(0)
                seq_id = self.llm_engine.add_request(**self._get_input_params(request, time.monotonic()))
                self.total_req += 1
                if isinstance(request, QARequest): 
                    self.rouge_dic[seq_id] = {"answer": request.answer}
                # logger.info(f"total request added : {self.total_req}")

            self._process_pending_sequences()
            step_outputs: List[RequestOutput] = self.llm_engine.step()
            self.num_steps += 1
            self._on_step_complete()

            for output in step_outputs:
                if output.finished:
                    self.num_processed_requests += 1
                    self.finished_seq(output)
                    if self.calculate_rouge:
                        self.rouge_dic[output.seq_id]["pred"] = [output.text.split('.')[0]]
                    # skip those with unknown tokens
                    if output.nll and self.use_logits:
                        self.num_token_processed += output.token_count
                        self.nll_sum += output.nll
                        t_logger.log(
                            key='instance.run',
                            message=f'nll_sum: {self.nll_sum} num token: {self.num_token_processed} perplexity: {torch.exp(self.nll_sum/self.num_token_processed)}'
                        )
                    t_logger.log(
                        key='instance.run',
                        message=f'num_processed_requests: {self.num_processed_requests}'
                    )
            self.gpu_max_blocks.set(self.llm_engine.config.cache_config.num_gpu_blocks)
            self.gpu_free_blocks.set(self.llm_engine.scheduler.block_manager.get_num_free_gpu_blocks())
        if self.calculate_rouge:
            self._evaluate_rouge()
        logger.debug('Returning from instance')
        self._run_finish.send.remote()
        self.run_loop_finished = True
        
    def _end_run(self):
        self.end_time = time.monotonic()
        logger.info(
            f"Instance {self.instance_id} exiting after processing {self.num_processed_requests} ({self.num_steps} iterations), Total time taken: {self.end_time - self.start_time:.2f} seconds"
        )
        if self.config.enable_profiling:
            self.llm_engine.stop_profiling()
        ray.get(self._run_finish.wait.remote())

    def _wrap_up_result(self):
        while not self.run_loop_finished: pass
        self._end_run()
        self.llm_engine.pull_worker_metrics()
        metric_store = self.llm_engine.get_metric_store()
        logger.info(f"instance {self.instance_id} wrapped up result and returned | {metric_store}")
        return metric_store

    def _clean_up(self):
        for worker in self.llm_engine.workers:
            ray.kill(worker)

    def run(self) -> None:
        logger.info(f'Warmming up instances {self.instance_id}')
        self.warmup()
        logger.info(f"Running instance {self.instance_id}")
        self.llm_engine.reset_metrics()
        self._prepare_run()
        self.ready_signal.send.remote()
        logger.info(f'Marking instance {self.instance_id} as ready')
        self._run()

    def _offline_inference(self, prompt, sampling_params):
        self.llm_engine.add_request(prompt, sampling_params)

    def _check_output_finished(self, expect_num_output):
        return len(self.output_list) == expect_num_output

    def _get_inference_result(self):
        return [PromptResponse(x.prompt, x.text) for x in self.output_list]

    def _get_running_status(self):
        return self._running

    def _sequence_finished_and_aborted(self):
        _ids = set(self._new_finished_ids)
        # self._new_finished_ids is not be thread-safe
        self._new_finished_ids -= _ids
        return {'total_count': len(self.output_list), 'ids': _ids}

    def _waiting_stop_signal(self):
        ray.get(self.stop_signal.wait.remote())
        self.terminate = True

    def _evaluate_rouge(self):
        references = []
        preds = []
        
        all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for each in self.rouge_dic.keys():
            references.append(self.rouge_dic[each]["answer"])
            preds.append(self.rouge_dic[each]["pred"])
        
        for ref, pred in zip(references, preds):
            scores = self.rouge.score(ref[0], pred[0])
            
            for rouge_type in all_scores:
                all_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        for rouge_type in all_scores:
            avg = sum(all_scores[rouge_type]) / len(all_scores[rouge_type])
            print(f"Avg {rouge_type} F1: {avg:.4f}")
