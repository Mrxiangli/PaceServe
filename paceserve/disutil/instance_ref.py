from dataclasses import dataclass

from paceserve.types import InstanceType


@dataclass
class InstanceRef:
    id: int
    _type: InstanceType
    ref: object
