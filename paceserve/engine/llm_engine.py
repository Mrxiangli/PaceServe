from paceserve.engine.base_llm_engine import BaseLLMEngine
from paceserve.config import SystemConfig, PaceSchedulerConfig, LowPrioritySchedulerConfig, DistserveSchedulerConfig, PrefillSchedulerConfig, DecodeSchedulerConfig
from paceserve.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine
from paceserve.engine.pace_llm_engine import PaceLLMEngine
from paceserve.engine.low_priority_llm_engine import LowPriorityLLMEngine
from paceserve.engine.prefill_llm_engine import PrefillLLMEngine
from paceserve.engine.decode_llm_engine import DecodeLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if config.parallel_config.pipeline_parallel_size > 1:
            engine = PipelineParallelLLMEngine(config)
        elif isinstance(config.scheduler_config, PaceSchedulerConfig):
            engine = PaceLLMEngine(config)
        elif isinstance(config.scheduler_config, LowPrioritySchedulerConfig):
            engine = LowPriorityLLMEngine(config)
        elif isinstance(config.scheduler_config, PrefillSchedulerConfig):
            engine = PrefillLLMEngine(config)
        elif isinstance(config.scheduler_config, DecodeSchedulerConfig):
            engine = DecodeLLMEngine(config)
        else:
            engine = BaseLLMEngine(config)
        return engine
