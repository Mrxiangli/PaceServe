import logging
import os
import time

import ray
from tqdm import tqdm
from multiprocessing import Process

from paceserve import LLMEngine, SamplingParams
from paceserve.benchmark.config import BenchmarkConfig
from paceserve.benchmark.entities import Request
from paceserve.benchmark.request_generator import RequestGeneratorRegistry
from paceserve.benchmark.utils.random import set_seeds
from paceserve.config import ReplicaConfig, ParallelConfig
from paceserve.config import InstanceConfig
from paceserve.types import ReplicaResourceMapping, ResourceMapping, InstanceResourceMapping
from paceserve.utils import get_ip
from paceserve.types import InstanceType
from paceserve.disutil.instance import Instance

logger = logging.getLogger(__name__)


@ray.remote
class MixInstance(Instance):

    def __init__(
        self,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
