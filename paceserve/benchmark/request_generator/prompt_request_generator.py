import logging
from typing import List

from paceserve.benchmark.entities.request import PromptRequest
from paceserve.benchmark.config import PromptRequestGeneratorConfig
from paceserve.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)

logger = logging.getLogger(__name__)


class PromptRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: PromptRequestGeneratorConfig):
        super().__init__(config)
        self.prompts = config.prompts

    def generate_requests(self) -> List[PromptRequest]:
        requests = []
        for prompt in self.prompts:
            request = PromptRequest(prompt=prompt)
            requests.append(request)
        return requests
