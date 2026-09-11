import logging
from typing import List

import ray

from paceserve.benchmark.config import BenchmarkConfig
from paceserve.types import ReplicaResourceMapping
from paceserve.utils import get_ip
from paceserve.utils.signal_actor import SignalActor
from paceserve.disutil.replica import Replica
from paceserve.core.datatypes.request_output import PromptResponse

logger = logging.getLogger(__name__)


class Launcher:

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.is_multi_replica = self.config.num_replicas > 1

        ray.init(ignore_reinit_error=True)

        self._validate_cluster_resources()
        self.replica_signals = []
        self.replicas = self._create_replicas()

    def _validate_cluster_resources(self):
        num_replicas = self.config.num_replicas
        num_instances = self.config.instance_config.num_instance
        num_cpus_per_instance = self.config.instance_config.num_cpus_per_instance

        self.gpus_per_replica = 0
        for idx in range(len(self.config.instance_config.instance_parallel)):
            tp,pp = self.config.instance_config.instance_parallel[idx]
            self.gpus_per_replica += tp*pp
        num_gpus_required = num_replicas * self.gpus_per_replica

        available_resources = ray.available_resources()
        available_gpus = available_resources["GPU"]
        available_cpus = available_resources["CPU"]

        assert (
            available_gpus >= num_gpus_required
        ), f"Insufficient GPUs. Required: {num_gpus_required}, Available: {available_gpus}"

        num_cpus_required = num_replicas * num_instances * num_cpus_per_instance
        assert (
            available_cpus >= num_cpus_required
        ), (
            f"Insufficient CPUs. Required: {num_cpus_required} "
            f"({num_replicas} replicas × {num_instances} instances × {num_cpus_per_instance} CPUs each). "
            f"Available: {available_cpus}. "
            f"Lower --instance_config_num_cpus_per_instance (currently {num_cpus_per_instance}) "
            f"or free up CPUs on the cluster."
        )

        logger.debug(f"Ray available resource in current cluster: {ray.available_resources()}")

    def _get_replica_resource_mapping(self) -> ReplicaResourceMapping:
        if self.config.replica_resource_mapping:
            assert len(self.config.replica_resource_mapping) == self.config.num_replicas
            logger.info(
                f"Replica resource mapping: {self.config.replica_resource_mapping}"
            )
            return self.config.replica_resource_mapping

        cluster_resources_keys = list(ray.available_resources().keys())
        num_gpus = ray.available_resources()["GPU"]
        ip_addresses = [
            x
            for x in cluster_resources_keys
            if x.startswith("node:") and x != "node:__internal_head__"
        ]
        logger.debug(f"#node & ip address: {ip_addresses}")

        replica_ip = f"node:{get_ip()}"

        ip_addresses.remove(replica_ip)
        ip_addresses.insert(0, replica_ip)

        num_nodes = len(ip_addresses)
        assert num_nodes > 0, "No nodes found in the cluster"
        assert num_gpus > 0, "No GPUs found in the cluster"
        assert (
            num_gpus % num_nodes == 0
        ), f"Number of GPUs ({num_gpus}) is not a multiple of number of nodes ({num_nodes})"
        num_gpus_per_node = int(num_gpus // num_nodes)
        num_replicas = self.config.num_replicas

        assert (
            num_gpus >= num_replicas *  self.gpus_per_replica  
        ), f"Insufficient GPUs. Required: {num_replicas *  self.gpus_per_replica}, Available: {num_gpus}"

        replica_resource_mapping = []

        available_gpus = []
        for ip_address in ip_addresses:
            for gpu_id in reversed(range(num_gpus_per_node)):
                available_gpus.append((ip_address, gpu_id))
                
        # here we evenly distribute the node resources to each replica
        for _ in range(num_replicas):
            resource_mapping = []
            for _ in range(self.gpus_per_replica):
                resource_mapping.append(available_gpus.pop(0))
            replica_resource_mapping.append(resource_mapping)

        logger.info(f"Replica resource mapping: {replica_resource_mapping}")

        return replica_resource_mapping

    def _create_replicas(self):
        replica_resource_mapping = self._get_replica_resource_mapping()
        replicas = []
        refs = []
        # the resource need to come from the same node, only passing
        # the resouce map, instance do the actual mapping
        for replica_id in range(self.config.num_replicas):
            finish_signal = SignalActor.remote()
            replica = Replica.options(
                    num_cpus=1,
                    max_concurrency = 2,
                    resources={
                        replica_resource_mapping[replica_id][0][0]: 0.01,
                    },
                ).remote(replica_id, replica_resource_mapping[replica_id],
                         self.config, finish_signal)
            refs.append(replica.setup.remote())
            replicas.append(replica)
            self.replica_signals.append(finish_signal)
        ray.get(refs)
        logger.info('Created replicas')
        return replicas

    def run(self):
        pending = {replica.run.remote(): replica for replica in self.replicas}
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1)
            for ref in ready:
                # Await the task itself so actor exceptions reach the driver.
                ray.get(ref)
                replica = pending.pop(ref)
                aggregate_metric = ray.get(replica._get_plot_handle.remote())
                aggregate_metric.plot()

    def get_prompt_responses(self) -> List[PromptResponse]:
        refs = []
        for replica in self.replicas:
            refs.append(replica.get_output_ids.remote())
        _reps_ids = ray.get(refs)
        return [x for _ids in _reps_ids for _idx in _ids for x in _idx]

    def cleanup(self):
        ray.shutdown()

    def stop(self):
        refs = []
        for replica in self.replicas:
            refs.append(replica.stop.remote())
        ray.get(refs)
