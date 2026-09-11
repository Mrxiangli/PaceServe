import logging
from typing import Tuple, Optional, List
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
from paceserve.benchmark.request_generator.base_request_generator import (
    BaseRequestGenerator,
)
from paceserve.benchmark.config import DatasetRequestGeneratorConfig
from paceserve.benchmark.entities.request import DatasetRequest, QARequest

logger = logging.getLogger(__name__)


class DatasetRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: DatasetRequestGeneratorConfig):
        super().__init__(config)
        self.dataset_name = config.dataset_name
        self.split = config.split
        self.max_requests = config.max_requests
        self.config_name = config.config_name
        self.individual_request = config.individual_request
        self.perplexity = config.perplexity
        # if not using individual request, we need to tokenize the entire dataset here and feed in 
        # overlapping chunks
        self.model_name = config.model_name
        if not self.individual_request:
            assert self.model_name != None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load dataset split internally
        self.dataset = load_dataset(self.dataset_name, self.config_name, split=self.split, trust_remote_code=True)
        # Optionally limit dataset size
        if config.max_requests is not None:
            print(f"config: max {config.max_requests}")
            self.dataset = self.dataset.select(range(min(len(self.dataset),self.max_requests)))

        self.next_request_idx = 0
    
    def construct_qa_prompt(self, example):
        question = example['question']
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].lower() + question[1:]
        context = example['context']
        prompt = (
            "You are a QA assistant. You will be given a context and a question."
            "Answer the question in no more than 5 words. DO NOT repate the question. DO NOT repeat the example. DO NOT make extra commentary.\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer: "
        )
        return prompt

    def generate_requests(self) -> List[DatasetRequest]:
        if self.individual_request:
            requests = []
            # for perplexity test
            if self.perplexity: 
                for example in self.dataset:
                    if len(example['text'])>0:
                        request = DatasetRequest(text=example['text'],prompt_ids=None)
                        requests.append(request)
                return requests
            else:   # for F1-score or other test
                for example in self.dataset:
                    prompt = self.construct_qa_prompt(example)
                    answer = example['answers']['text']
                    requests.append(QARequest(text=prompt, prompt_ids=None, answer=answer))

                return requests
        
        else:
            all_text = "\n\n".join(self.dataset["text"])
            tokenized = self.tokenizer(all_text, return_tensors=None, truncation=False)
            input_ids = tokenized["input_ids"]

            seq_len = 1024
            stride = 512

            requests = []
            for i in range(0, len(input_ids) - seq_len, stride):
                chunk_ids = input_ids[i:i+seq_len]
                requests.append(DatasetRequest(text="",prompt_ids=chunk_ids))
                
            return requests   
