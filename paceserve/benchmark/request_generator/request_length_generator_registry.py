from paceserve.benchmark.request_generator.fixed_request_length_generator import (
    FixedRequestLengthGenerator,
)
from paceserve.benchmark.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from paceserve.benchmark.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from paceserve.benchmark.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)
from paceserve.types import RequestLengthGeneratorType
from paceserve.utils.base_registry import BaseRegistry


class RequestLengthGeneratorRegistry(BaseRegistry):
    pass


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.FIXED, FixedRequestLengthGenerator
)
