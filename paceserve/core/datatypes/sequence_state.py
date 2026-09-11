import time
import torch
from typing import Optional

from paceserve.core.datatypes.sequence_status import SequenceStatus

absolute_time = lambda: time.time()


class SequenceState:

    def __init__(self, id: str, arrived_at: float, num_prompt_tokens: int):
        self._id = id
        self._arrived_at: float = arrived_at
        self._left_prefill_at: Optional[float] = None
        self._arrived_decode_at: Optional[float] = None
        self._reschedule_start: Optional[float] = None
        self._reschedule_end: Optional[float] = None
        self._seq_left_at: Optional[float] = None
        self._seq_arrive_at: Optional[float] = None
        self._num_prompt_tokens: int = num_prompt_tokens
        self._num_output_tokens: int = 0
        self._status = SequenceStatus.WAITING
        self._is_scheduled: bool = False
        self._is_completed: bool = False
        self._scheduled_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._prompt_processing_completed_at: Optional[float] = None
        self._last_restart_at: Optional[float] = None
        self._last_pause_at: Optional[float] = None
        self._execution_time: float = 0.0
        self._preempted_time: float = 0.0
        self._paused_time: float = 0.0
        self._last_execution_start_at: Optional[float] = None
        self._num_restarts: int = 0
        self._num_pauses: int = 0
        self._is_ignore_finished: bool = False
        self._last_token_generated_at: Optional[float] = None
        self._last_token_generation_time: float = 0.0
        self._kv_transfer_started_at: Optional[float] = None
        self._kv_transfer_finished_at: Optional[float] = None
        self._request_reschedule_interval: Optional[float] = 0.0
        self._request_aborted: Optional[int] = 0
        self._estimated_prefill_time: float = 0.0
        self._wait_time: float = 0.0
        self._preemption_count: int = 0
        self._request_offloaded_time: Optional[float] = None

        self._absolute_arrived_at : float = absolute_time()
        self._absolute_finished_at: float = -1.0
        self._seq_perplexity: float = 0
        self._seq_nll: float = None
        self._seq_token_count: int = None
        # when metadata left prefill
        self._kvmeta_left_prefill: float = None
        self._kvmeta_left_prefill_monotonic: float = None
        # when metadata arrive decode
        self._kvmeta_arrive_decode: float = None
        # when meta data get scheduled on decode machine
        self._kvmeta_get_scheduled: float = None
        # new seq kv insertion timestamp
        self._kv_insertion_start: float = None
        # new seq kv insertion timestamp
        self._kv_insertion_finish: float = None
        # worker kv_insert 
        self._worker_kv_insertion: float = None
        # worker kv_allocate 
        self._worker_kv_allocate: float = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def num_prompt_tokens(self) -> int:
        return self._num_prompt_tokens

    @property
    def num_output_tokens(self) -> int:
        return self._num_output_tokens

    @property
    def num_total_tokens(self) -> int:
        return self._num_prompt_tokens + self._num_output_tokens

    @property
    def status(self) -> SequenceStatus:
        return self._status

    @property
    def is_scheduled(self) -> bool:
        return self._is_scheduled

    @property
    def is_completed(self) -> bool:
        return self._is_completed

    @property
    def arrived_at(self) -> float:
        return self._arrived_at
    
    @property
    def kvmeta_idle_time(self) -> float:
        # both are time.monotonic from the prefill machine
        if self._kvmeta_left_prefill_monotonic:
            return self._kvmeta_left_prefill_monotonic - self._prompt_processing_completed_at
        else:
            return None
    
    @property
    def kvmeta_transfer_time(self) -> float:
        if  self._kvmeta_arrive_decode:
            return  self._kvmeta_arrive_decode  - self._kvmeta_left_prefill
        else:
            return None
    
    @property
    def kvmeta_schedule_time(self) -> float:
        if  self._kvmeta_get_scheduled:
            return  self._kvmeta_get_scheduled  - self._kvmeta_arrive_decode
        else:
            return None
    
    @property
    def kv_insertion_time(self) -> float:
        if self._kv_insertion_finish:
            return self._kv_insertion_finish - self._kv_insertion_start
        else:
            return None
    
    @property
    def worker_kv_insertion_time(self) -> float:
        if self._worker_kv_insertion:
            return self._worker_kv_insertion
        else:
            return None
    
    @property
    def worker_kv_allocate_time(self) -> float:
        if self._worker_kv_allocate:
            return self._worker_kv_allocate
        else:
            return None
    
    @property
    def decode_tbt_mean(self) -> float:
        if self._completed_at:
            if not self._kv_transfer_finished_at:    # no disaggregation
                return 0 if self._num_output_tokens ==1 else (self._completed_at - self._prompt_processing_completed_at) / (self._num_output_tokens -1)
            else:
                return 0 if self._num_output_tokens ==1 else (self._completed_at - self._arrived_decode_at) / (self._num_output_tokens -1)
                
    @property
    def scheduled_at(self) -> Optional[float]:
        return self._scheduled_at
    
    @property
    def request_preemption_count(self) -> bool:
        return self._preemption_count

    @property
    def completed_at(self) -> Optional[float]:
        return self._completed_at

    @property
    def high_priority_scheduling_delay(self) -> Optional[float]:
        if self._seq_arrive_at:
            return self.scheduled_at - self._seq_arrive_at
        else:
            return 0

    @property
    def prompt_processing_completed_at(self) -> Optional[float]:
        return self._prompt_processing_completed_at

    @property
    def e2e_time(self) -> Optional[float]:
        # prefill and decode on different machine
        # self._left_prefill_at - self._arrived_at are measured  with time.monotonic on prefill machine
        if self._arrived_decode_at: 
            return (
                (self._left_prefill_at - self._arrived_at) + (self._completed_at - self._arrived_decode_at) + (self._kv_transfer_finished_at - self._kv_transfer_started_at)
                if self._completed_at is not None
                else None
            )
        # sequence is rescheduled on a different machine
        elif self._seq_left_at:
            return (
                (self._seq_left_at - self.arrived_at) + (self._reschedule_end - self._reschedule_start) + (self._completed_at - self._seq_arrive_at)
                if self._completed_at is not None
                else None
            )
        else:
            return (
                self._completed_at - self._arrived_at
                if self._completed_at is not None
                else None
            )
    
    @property
    def reschedule_idle(self) -> Optional[float]:
        if self._reschedule_start:
            self._request_reschedule_interval = (self._seq_left_at - self.arrived_at) + (self._reschedule_end - self._reschedule_start)
        else:
            self._request_reschedule_interval = 0.0
            
        return self._request_reschedule_interval
    
    @property
    def waiting_time(self) -> Optional[float]:
        if self._reschedule_start and self._reschedule_end:
            self._wait_time = self.reschedule_idle + time.monotonic() - self._seq_arrive_at
        else:
            self._wait_time = time.monotonic() - self.arrived_at
            
        return self._wait_time
            
    @property
    def e2e_time_piecewise_normalized(self) -> float:
        return self.scheduling_delay + (
            self.execution_plus_preemption_time / self._num_output_tokens
        )

    @property
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self._num_output_tokens

    @property
    def e2e_prefill_time(self) -> Optional[float]:
        # if there is reschedule
        if self._seq_left_at:
            return (
                (self._prompt_processing_completed_at - self._seq_arrive_at) + (self._reschedule_end - self._reschedule_start) \
                                                                                        + (self._seq_left_at - self.arrived_at)
                if self._prompt_processing_completed_at is not None
                else None
            )
        else:
            return (
                self._prompt_processing_completed_at - self._arrived_at
                if self._prompt_processing_completed_at is not None
                else None
            )

    @property
    def e2e_prefill_time_normalized(self) -> Optional[float]:
        return (
            (self.e2e_prefill_time / self._num_prompt_tokens)
            if self._prompt_processing_completed_at is not None
            else None
        )

    @property
    def e2e_prefill_time_piecewise_normalized(self) -> Optional[float]:
        return (
            self.scheduling_delay
            + (self.prefill_execution_plus_preemption_time / self._num_prompt_tokens)
            if self._prompt_processing_completed_at
            else None
        )

    @property
    def prefill_execution_plus_preemption_time(self) -> float:
        return (
            self._prompt_processing_completed_at - self._scheduled_at
            if self._prompt_processing_completed_at is not None
            else None
        )

    @property
    def estimated_prefill_time(self) -> float:
        return self._estimated_prefill_time

    @property
    def decode_execution_plus_preemption_time(self) -> float:
        # on single machine
        if not self._arrived_decode_at:
            return (
                self._completed_at - self._prompt_processing_completed_at
                if self._completed_at is not None
                else None
            )
        else:
            return (
                self._completed_at - self._arrived_decode_at
                if self._completed_at is not None
                else None
            )

    @property
    def prefill_execution_plus_preemption_time_normalized(self) -> Optional[float]:
        return (
            self.prefill_execution_plus_preemption_time / self._num_prompt_tokens
            if self.prefill_execution_plus_preemption_time
            else None
        )

    @property
    def decode_execution_plus_preemption_time_normalized(self) -> Optional[float]:
        return (
            self.decode_execution_plus_preemption_time / self._num_output_tokens
            if self.decode_execution_plus_preemption_time
            else None
        )

    @property
    def scheduling_delay(self) -> Optional[float]:
        # if there is sequence offloading
        if self._seq_arrive_at:
            return(
                (self.scheduled_at - self._seq_arrive_at) + (self._reschedule_end - self._reschedule_start) + (self._seq_left_at - self._arrived_at)
            )
        else:
            return (
                self._scheduled_at - self._arrived_at
                if self._scheduled_at is not None
                else None
            )

    @property
    def execution_time(self) -> float:
        return self._execution_time

    @property
    def execution_time_normalized(self) -> float:
        return self.execution_time / self._num_output_tokens

    @property
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    def paused_time(self) -> float:
        return self._paused_time

    @property
    def execution_plus_preemption_time(self) -> float:
        return self._execution_time + self._preempted_time

    @property
    def execution_plus_preemption_time_normalized(self) -> float:
        return self.execution_plus_preemption_time / self._num_output_tokens

    @property
    def last_token_generation_time(self) -> float:
        return self._last_token_generation_time

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def num_pauses(self) -> int:
        return self._num_pauses

    @property
    def is_ignore_finished(self) -> bool:
        return self._is_ignore_finished

    @property
    # this includes the time from the sequence kv cache being pull out of the prefill memory, 
    # untill its being inserted into the decode GPU memory; the prefill_kv_idle time includes other
    # time that the request spend waiting on the prefill machine due to decode machien lack of memory
    def kv_transfer_network_time(self) -> float:
        if self._kv_transfer_started_at and self._kv_transfer_finished_at:
            return self._kv_transfer_finished_at - self._kv_transfer_started_at
        else:
            return 0
    @property
    def request_reschedule_interval(self) -> float:
        if self._reschedule_end:
            return self._reschedule_end - self._reschedule_start
        return 0
    
    @property
    def request_aborted(self) -> bool:
        return self._request_aborted
    
    @property
    def request_offloaded_time(self) -> float:
        return self._request_offloaded_time

    @property
    def absolute_arrived_at(self) -> float:
        return self._absolute_arrived_at

    @property
    def absolute_finished_at(self) -> float:
        return self._absolute_finished_at

    @property
    def seq_perplexity(self) -> float:
        return self._seq_perplexity

    def _handle_transitions_from_waiting_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        if status == SequenceStatus.RUNNING:
            # request is starting execution now
            if self._scheduled_at is None:
                # running for the first time
                assert self._num_restarts == 0
                self._is_scheduled = True
                self._scheduled_at = current_time
            else:
                # restarting
                assert self._num_restarts > 0
                self._preempted_time += current_time - self._last_restart_at

            self._last_execution_start_at = current_time
        elif status == SequenceStatus.FINISHED_IGNORED:
            self._is_ignore_finished = True
            self._is_completed = True
            self._completed_at = current_time
            # the scheduler will not schedule this request again
            self._scheduled_at = current_time
            self._absolute_finished_at = absolute_time()
        # TODO: Think about merging these two
        elif status == SequenceStatus.FINISHED_ABORTED:
            self._request_aborted = 1
            self._is_completed = True
            self._completed_at = current_time
            # the scheduler will not schedule this request again
            self._scheduled_at = current_time
            self._absolute_finished_at = absolute_time()
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_running_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        self._execution_time += current_time - self._last_execution_start_at

        if status == SequenceStatus.PAUSED:
            self._num_pauses += 1
            self._last_pause_at = current_time
        elif status == SequenceStatus.WAITING:
            self._num_restarts += 1
            self._last_restart_at = current_time
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def _handle_transitions_from_paused_status(
        self, current_time: float, status: SequenceStatus
    ) -> None:
        self._paused_time += current_time - self._last_pause_at
        self._preempted_time += current_time - self._last_pause_at

        if (
            status == SequenceStatus.FINISHED_STOPPED
            or status == SequenceStatus.FINISHED_LENGTH_CAPPED
        ):
            self._is_completed = True
            self._completed_at = current_time
            self._absolute_finished_at = absolute_time()
        elif status == SequenceStatus.RUNNING:
            self._last_execution_start_at = current_time
        elif status == SequenceStatus.WAITING:
            self._num_restarts += 1
            self._last_restart_at = current_time
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

    def set_status(self, status: SequenceStatus) -> None:
        current_time = time.monotonic()

        if self._status == SequenceStatus.WAITING:
            self._handle_transitions_from_waiting_status(current_time, status)
        elif self._status == SequenceStatus.RUNNING:
            self._handle_transitions_from_running_status(current_time, status)
        elif self._status == SequenceStatus.PAUSED:
            self._handle_transitions_from_paused_status(current_time, status)
        else:
            raise ValueError(
                f"Invalid state transition from {self._status} to {status} for request {self._id}."
            )

        self._status = status

    def on_prompt_processing_completed(self) -> None:
        self._prompt_processing_completed_at = time.monotonic()
    
    def on_kv_insertion_finished(self) -> None:
        self._kv_insertion_finish = time.monotonic()
        self._last_token_generated_at = time.monotonic() - (self._kv_transfer_finished_at-self._kv_transfer_started_at)
        self._arrived_decode_at = time.monotonic()
    
    def on_request_abortion(self) -> None:
        self._request_aborted = 1
        
    def on_reschedule_start(self) -> None:
        self._reschedule_start = time.time()
    
    def on_reschedule_end(self) -> None:
        self._reschedule_end = time.time()
    
    def on_seq_left(self) -> None:
        self._seq_left_at = time.monotonic()
        
    def on_seq_arrive(self) -> None:
        self._seq_arrive_at = time.monotonic()
    
    def on_offload(self) -> None:
        self._request_offloaded_time = time.monotonic() - self._arrived_at
    
    def on_preemption(self) -> None:
        self._preemption_count += 1

    def on_token_generated(self) -> None:
        current_time = time.monotonic()

        self._num_output_tokens += 1

        if not self._last_token_generated_at:
            self._last_token_generation_time = 0
        else:
            self._last_token_generation_time = (
                current_time - self._last_token_generated_at
            )

        self._last_token_generated_at = current_time
