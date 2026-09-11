from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

GPULocation = Tuple[Optional[str], int]  # (node_ip, gpu_id)
ResourceMapping = List[GPULocation]
ReplicaResourceMapping = List[ResourceMapping]  # List ResourceMapping for each replica
InstanceResourceMapping = List[ResourceMapping]  # List ResourceMapping for each replica

class SchedulerType(Enum):
    DIST = 'DIST'
    PACE = 'PACE'
    HIGH = 'HIGH'
    LOW = 'LOW'
    VLLM = "VLLM"
    ORCA = "ORCA"
    FASTER_TRANSFORMER = "FASTER_TRANSFORMER"
    SARATHI = "SARATHI"
    SIMPLE_CHUNKING = "SIMPLE_CHUNKING"
    NIYAMA = 'NIYAMA'
    PREFILL = 'PREFILL'
    DECODE = 'DECODE'


class RequestGeneratorType(Enum):
    SYNTHETIC = "SYNTHETIC"
    TRACE = "TRACE"
    PROMPTS = "PROMPTS"
    DATASET = "DATASET"


class RequestIntervalGeneratorType(Enum):
    POISSON = "POISSON"
    GAMMA = "GAMMA"
    STATIC = "STATIC"
    TRACE = "TRACE"


class RequestLengthGeneratorType(Enum):
    UNIFORM = "UNIFORM"
    ZIPF = "ZIPF"
    TRACE = "TRACE"
    FIXED = "FIXED"


class AttentionBackend(Enum):
    FLASHINFER = "FLASHINFER"
    NO_OP = "NO_OP"
    NAIVE = "NAIVE"


class InstanceType(Enum):
    PREFILL = 1002
    DECODE = 2002
    MIX = 3002
    HIGH_PRIORITY = 10001
    LOW_PRIORITY = 10002

class ControllerType(Enum):
    ROUNDROBIN = "roundrobin"
    DISTSERVE = "distserve"
    PACE = "pace"