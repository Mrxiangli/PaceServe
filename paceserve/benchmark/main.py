import os
import yaml
import logging
from threading import Thread
import time
from typing import List

from paceserve.benchmark.launcher import Launcher
from paceserve.benchmark.config import BenchmarkConfig, \
    PromptRequestGeneratorConfig, DatasetRequestGeneratorConfig
from paceserve.benchmark.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from paceserve.benchmark.utils.random import set_seeds
from paceserve.logger import init_logger
from paceserve.types import RequestGeneratorType
from paceserve.core.datatypes.request_output import PromptResponse


from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

# This is the initial point and must pass the name of the
# root logger.
logger = init_logger('paceserve')


PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def sanitize_yaml(obj):
    """
    Recursively convert all non-serializable objects (like Enums) into strings,
    ensure single-item lists are flattened, and prepare the config for YAML dumping.
    """
    if isinstance(obj, list):
        # If the list has only one element and it's a basic type, extract it
        if len(obj) == 1 and isinstance(obj[0], (str, int, float, bool)):
            return obj[0]
        return [sanitize_yaml(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool)):  
        return obj  # Keep basic types unchanged
    elif hasattr(obj, "name"):  
        return obj.name  # Convert Enum objects to their string names
    else:
        return str(obj)  # Convert any unknown object to string

def main() -> None:
    logger.info('Starting the main...')
    config: BenchmarkConfig = BenchmarkConfig.create_from_cli_args()
    use_prompts = config.request_generator_config.get_type() == RequestGeneratorType.PROMPTS
    if use_prompts:
        config.request_generator_config = PromptRequestGeneratorConfig(prompts=PROMPTS)
    
    use_dataset = config.request_generator_config.get_type() == RequestGeneratorType.DATASET
    if use_dataset:
        print("USE DATASET")
        #config.request_generator_config = DatasetRequestGeneratorConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
        config_dict = config.to_dict()
        simplified_config = sanitize_yaml(config_dict)
        yaml.safe_dump(simplified_config, f, default_flow_style=False)
    logger.debug(f"Starting benchmark with config: {config}")
    set_seeds(config.seed)

    log_level = getattr(logging, config.log_level.upper())
    logging.basicConfig(
        format=LOGGER_FORMAT, level=log_level, datefmt=LOGGER_TIME_FORMAT
    )
    def timer_loop(limit):
        start = time.time()
        while True:
            time.sleep(1)
            if time.time() - start > limit:
                logger.info('============ main is stopping the launcher')
                launcher.stop()
                break

    # creating launcher
    launcher = Launcher(config)
    time_count = Thread(target=timer_loop, args=((config.warmup_time + config.time_limit) * 20,), daemon=True)
    time_count.start()
    launcher.run()

    if use_prompts:
        outputs: List[PromptResponse] = launcher.get_prompt_responses()
        for out in outputs:
            print(f'{out.prompt} | {out.response}')
    
    if use_dataset and config.request_generator_config.individual_request:
        outputs: List[PromptResponse] = launcher.get_prompt_responses()
        for out in outputs:
            print(f"==========================")
            print(f'{out.prompt[:200]}')
            print(f"Answer: {out.response.split('.')[0]} \n")

    print(f"result at: {config.output_dir}")

if __name__ == "__main__":
    main()
