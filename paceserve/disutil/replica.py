import ray
import logging
import os
import time
import copy

from paceserve.benchmark.config import BenchmarkConfig
from paceserve.config import ReplicaConfig, ParallelConfig
from paceserve.types import ReplicaResourceMapping, InstanceResourceMapping
from paceserve.utils import get_ip
from paceserve.metrics.metrics_store import MetricsStore

from paceserve.core.controller.controller_registry import ControllerRegistry
from paceserve.core.controller.roundrobin_controller import RoundRobinController
from paceserve.core.controller.distserve_controller import DistServeController
from paceserve.disutil.instance import Instance
from paceserve.disutil.mix_instance import MixInstance
from paceserve.disutil.prefill_instance import PrefillInstance
from paceserve.disutil.decode_instance import DecodeInstance
from paceserve.disutil.hp_instance import HPInstance
from paceserve.disutil.lp_instance import LPInstance
from paceserve.disutil.instance_ref import InstanceRef
from paceserve.types import InstanceType
from paceserve.utils.signal_actor import SignalActor


INSTANCE_CLASS = {
    InstanceType.PREFILL: PrefillInstance,
    InstanceType.DECODE: DecodeInstance,
    InstanceType.MIX: MixInstance,
    InstanceType.HIGH_PRIORITY: HPInstance,
    InstanceType.LOW_PRIORITY: LPInstance
}

logger = logging.getLogger(__name__)


@ray.remote
class Replica:
    def __init__(self, replica_id: int, 
                 resource_mapping: ReplicaResourceMapping, 
                 config: BenchmarkConfig,
                 finish_signal: SignalActor) -> None:
        self.id = replica_id
        self.config = config
        self.resource_mapping = resource_mapping
        self.finish_signal = finish_signal

    def get_output_ids(self):
        return ray.get(self.controller.get_output_ids.remote())

    def setup(self):
        self._validate_replica_resources()
        self.instances, self.ready_signals, self.stop_signal = self._create_instances()
        _i_refs = self._setup_instances()
        _c_refs = self._create_controller()

        self.aggregate_metric_store = self._create_aggregate_metric_store()
        self._wait_with_instances(_c_refs + _i_refs)
        logger.debug('Created controller, instances and populated metric store.')

    def _create_controller(self):
        self.controller_stop_signal = SignalActor.remote()
        self.controller = ControllerRegistry.get_ray_actor(
            self.config.controller_config.get_type(),
            self.config,
            self.instances,
            self.controller_stop_signal,
            self.stop_signal,
            num_cpus=12,
            max_concurrency=8
        )

        return [self.controller.generate_request.remote()]


    def _get_plot_handle(self):
        return self.aggregate_metric_store

    def _validate_replica_resources(self):
        num_instance = self.config.instance_config.num_instance
        self.num_gpus_required = 0
        for idx in range(len(self.config.instance_config.instance_parallel)):
            tp,pp = self.config.instance_config.instance_parallel[idx]
            self.num_gpus_required += tp*pp
            print(f"replica {self.id} instance {idx} gpu required: {tp*pp}")

        available_gpus = len(self.resource_mapping)
        assert (
            available_gpus >= self.num_gpus_required
        ), f"Insufficient GPUs. Required: {self.num_gpus_required}, Available: {available_gpus}"

    def _get_instance_resource_mapping(self) -> InstanceResourceMapping:
        # num_gpus = len(self.resource_mapping)
        logger.info(f"current replica resource: {self.resource_mapping}")
        instance_resource_mapping = []
        for each_config in self.config.instance_config.instance_parallel:
            resource_mapping = []
            tp,pp = each_config
            for _ in range(tp*pp):
                # need to add check to make sure all GPUs on one node
                resource_mapping.append(self.resource_mapping.pop(0))   
            instance_resource_mapping.append(resource_mapping)
        logger.info(f"Instance resource mapping: {instance_resource_mapping}")
        return instance_resource_mapping

    def _create_instances(self):
        instance_resource_mapping = self._get_instance_resource_mapping()

        instances = []
        ready_signals = []
        stop_signal = SignalActor.remote()
        for i, i_type in enumerate(self.config.instance_config.instances):
            _type = InstanceType[i_type.upper()]
            _ready_signal = SignalActor.remote()

            tp, pp = self.config.instance_config.instance_parallel[i]
            instance_parallel_config = ParallelConfig(pp, tp)
            instance_config = copy.deepcopy(self.config.instance_config)
            instance_config._type = _type
            instance_config.resource_mapping = instance_resource_mapping[i]
            new_inst = INSTANCE_CLASS[_type].options(
                num_cpus=self.config.instance_config.num_cpus_per_instance,
                resources={
                    instance_resource_mapping[i][0][0]: 0.01,
                },
                max_concurrency = 18,
                ).remote(self.id, i, self.config, instance_parallel_config, instance_config, _ready_signal, stop_signal, self.config.controller_config.use_logits)
            new_inst._set_ref.remote(new_inst)
            instances.append(InstanceRef(i, _type, new_inst))
            ready_signals.append(_ready_signal)
        return instances, ready_signals, stop_signal

    def _create_aggregate_metric_store(self):
        replica_config = ReplicaConfig(
            replica_id=self.id, 
            output_dir=self.config.output_dir,
        )
        # We only create MetricStore in replica so later we can merge this
        # instance with the metric stores from the instances. This 
        # metric store does not log anything.
        metrics_store = MetricsStore.get_or_create_instance(
            replica_config,
            self.config.model_config,
            self.config.metrics_config,
        )

        metrics_store.mark_initial_memory_profiling_done()

        return metrics_store

    def _setup_instances(self):
        # Run all the instances
        self.instance_run_refs = [
            instance.ref.run.remote() for instance in self.instances
        ]
        # Wait for all instances to be ready
        return [_ready.wait.remote() for _ready in self.ready_signals]
    
    def _wait_with_instances(self, refs):
        pending = list(refs)
        while pending:
            ready, _ = ray.wait(
                pending + self.instance_run_refs, num_returns=1
            )
            for ref in ready:
                # Propagate engine failures instead of waiting on their signals.
                ray.get(ref)
                if ref in self.instance_run_refs:
                    self.instance_run_refs.remove(ref)
                if ref in pending:
                    pending.remove(ref)

    def run(self):
        # Monitor engine failures while the controller waits for completions.
        self._wait_with_instances([self.controller.schedule_requests.remote()])
        ray.get(self.instance_run_refs)

        # get the metric stores from the instances and merge it into one

        _int_metric_stores = []
        for _int in self.instances:
            data = ray.get(_int.ref._wrap_up_result.remote())
            _int_metric_stores.append(data)
            
        for data in _int_metric_stores:
            self.aggregate_metric_store.merge(data)

        # Send the signal to launcher that this replica is done
        self.finish_signal.send.remote()
    
    def stop(self):
        ray.get(self.controller_stop_signal.send.remote())
