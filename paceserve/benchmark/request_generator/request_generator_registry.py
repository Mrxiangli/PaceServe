from paceserve.benchmark.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from paceserve.benchmark.request_generator.trace_request_generator import (
    TraceRequestGenerator,
)
from paceserve.benchmark.request_generator.prompt_request_generator import (
    PromptRequestGenerator,
)
from paceserve.benchmark.request_generator.dataset_request_generator import (
    DatasetRequestGenerator,
)
from paceserve.types import RequestGeneratorType
from paceserve.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
RequestGeneratorRegistry.register(RequestGeneratorType.PROMPTS, PromptRequestGenerator)
RequestGeneratorRegistry.register(RequestGeneratorType.DATASET, DatasetRequestGenerator)
