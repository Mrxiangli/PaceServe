import json
from typing import Any, Dict, List

from paceserve.core.datatypes.batch import Batch
from paceserve.core.datatypes.sequence import Sequence
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs


INSTANCE_TID = 0
WORKER_TID = 1

SCHEDULER_CAT = 'scheduler'
WORKER_CAT = 'worker'


def _order_seqs(seqs: List[Sequence]):
    return sorted(seqs, key=lambda _seq: _seq.seq_id)


class Tracer:
    # This class provides the following hooks:
    # `on_schedule` is called on the scheduler when a batch is prepared
    # `on_batch_stage_end` is called on a worker when a batch finishes
    # its execution.
    # `on_batch_end` is called on the scheduler when the batch finishes
    # and the scheduler gets the response from all the workers.
    # NOTE: For all the hooks, the start_time and end_time must be given
    # in the unit of seconds.
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.chrome_trace: List[Dict[str, Any]] = []

    def store_traces(self, out_dir):
        file_path = f"{out_dir}/tracer_chrome_trace.json"
        with open(file_path, "w") as f:
            json.dump(self.chrome_trace, f)

    def merge(self, other: "Tracer"):
        self.chrome_trace.extend(other.chrome_trace)

    def on_schedule(
        self,
        scheduler_metadata: SchedulerMetadata,
        start_time: float,
        end_time: float
    ):
        batch_id = scheduler_metadata.id
        batch: Batch = scheduler_metadata.batch
        batch.sort_requests()
        items = sorted(scheduler_metadata.items, key=lambda item: item.sequence.seq_id)
        args = {
            'batch_size': batch.size,
            'running': [str(x) for x in batch.get_running_requests()],
            'waiting': [str(x) for x in _order_seqs(scheduler_metadata.waiting)],
            'preempted': [str(x) for x in _order_seqs(scheduler_metadata.preempted)],
            'partial_prefills': [str(x) for x in _order_seqs(scheduler_metadata.partial_prefills)],
            'items': [str(x) for x in items],
            'seq_transfer_list': [str(x) for x in _order_seqs(scheduler_metadata.seq_transfer_list)]
        }
        trace = self._to_chrome_trace_dict(batch_id, start_time, end_time,
                                           SCHEDULER_CAT, INSTANCE_TID, args=args)
        self.chrome_trace.append(trace)

    def on_batch_stage_end(
        self,
        scheduler_outputs: SchedulerOutputs,
        start_time: float,
        end_time: float
    ):
        batch_id = scheduler_outputs.id
        seq_ids = scheduler_outputs.seq_ids
        args = {
            'batch_size': len(seq_ids),
            'seq_ids': seq_ids,
        }
        trace = self._to_chrome_trace_dict(batch_id, start_time, end_time,
                                           WORKER_CAT, WORKER_TID, args=args)
        self.chrome_trace.append(trace)

    def on_batch_end(
        self,
        scheduler_outputs: SchedulerOutputs,
        start_time: float,
        end_time: float
    ):
        pass

    def _to_chrome_trace_dict(
        self,
        name: str,
        start_time: float,
        end_time: float,
        cat: str,
        tid: int,
        args: Dict={}
    ):
        return {
            'name': str(name),
            'ph': 'X',
            'cat': cat,
            'ts': start_time * 1e6,
            'dur': (end_time - start_time) * 1e6,
            'pid': self.instance_id,
            'tid': tid,
            'args': args
        }
