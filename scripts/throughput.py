"""Benchmark offline inference throughput."""
import dataclasses
from typing import List, Optional
from datasets import DatasetDict

@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        multi_modal_data: Optional dictionary containing multi-modal data (e.g.
            images).
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
    """
    prompt: str
    prompt_len: int
    expected_output_len: int


def sample_requests(inp_ds: DatasetDict,
                    target_ds: DatasetDict,
                    max_length: int,
                    fixed_output_len: Optional[int] = None,
    	            ):
    
    # Filter out sequences that are too long or too short
    filtered_dataset: List[SampleRequest] = []
    for inputs, target in zip(inp_ds['test'], target_ds['test']):

        inputs = inputs['input_ids']
        target = target['input_ids']

        prompt_len = len(inputs)
        output_len = len(target
                         ) if fixed_output_len is None else fixed_output_len
        
        if (prompt_len + output_len) > max_length:
            continue

        filtered_dataset.append(
            SampleRequest(prompt=inputs,
                          prompt_len=prompt_len,
                          expected_output_len=output_len,
                        )
        )

    return filtered_dataset