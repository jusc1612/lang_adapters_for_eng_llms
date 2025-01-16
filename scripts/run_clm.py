# this script builds on top of https://github.com/adapter-hub/adapters/blob/main/examples/pytorch/language-modeling/run_clm.py

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import os
active_env = os.getenv('CONDA_DEFAULT_ENV')
if "adapters" in active_env:
    from adapters import (
        AutoAdapterModel, 
        AdapterArguments, 
        AdapterConfig, 
        AdapterFusionConfig,
        AdapterTrainer
    )
    from adapters.composition import Stack
    import adapters
    from setup_adapter_training import setup_adapter_training

elif active_env == "peft":
    from peft import (
    get_peft_model,
    get_peft_config,  
    PromptTuningInit, 
    )

import wandb
import ast
import logging
import math
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Tuple, Dict
import utils
from utils import main_process_print, get_hf_dataset_files, get_language_mapping, code_2_lang

import datasets
import torch
from datasets import load_dataset, Dataset, concatenate_datasets

import evaluate
import torch.distributed
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint, seed_worker
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.utils.import_utils import is_datasets_available

import numpy as np
from collections import defaultdict


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            metrics["perplexity"] = np.exp(metrics["eval_loss"])
            print(f"Perplexity: {metrics['perplexity']}")

class MultilingualDataset(Dataset):
    def __init__(self, language_dict: dict, language_ratios: dict, batch_size: int):
        """
        Initialize a multilingual dataset with a dictionary of preprocessed (tokenized and chunked) datasets for each language.
        Args:
            language_dict (dict): A dictionary where the key is the language name 
                                  and the value is a tokenized HuggingFace dataset.
            language_ratios (dict): A dictionary with language as key and ratio of that language
                                    in dataset.
            batch_size (int): Batch size per device.
        """
        self.language_dict = language_dict
        self.languages = list(language_dict.keys())
        self.language_ratios = language_ratios
        self.language_iters = {lang: iter(language_dict[lang]) for lang in self.languages}
        self.batch_size = batch_size

        # Calculate total number of batches for each language
        self.language_batch_counts = {}
        for lang, dataset in language_dict.items():
            self.language_batch_counts[lang] = len(dataset) // batch_size

        self.language_steps = {lang: int(ratio * 10) for lang, ratio in language_ratios.items()}
        self.current_step_count = 0
        self.current_language_index = 0
    
    def __len__(self):
        # Calculate the length as the total number of batches across all languages
        #return sum(len(dataset) for dataset in self.language_dict.values())
        return sum(self.language_batch_counts.values())
    
    def __getitem__(self, idx):
        """
        Returns a batch from the current language, then moves to the next language.
        """
        current_language = self.languages[self.current_language_index]
        max_steps = self.language_steps[current_language]
        
        # Get the next batch from the current language dataset
        try:
            #batch = next(self.language_iters[current_language])
            batch = [next(self.language_iters[current_language]) for _ in range(self.batch_size)]
        except StopIteration:
            # If the iterator is exhausted, reset it and take the first batch again
            self.language_iters[current_language] = iter(self.language_dict[current_language])
            #batch = next(self.language_iters[current_language])
            batch = [next(self.language_iters[current_language]) for _ in range(self.batch_size)]
        
        self.current_step_count += 1

        if self.current_step_count >= max_steps:
            self.current_step_count = 0
            self.current_language_index = (self.current_language_index + 1) % len(self.languages)
        
        return batch

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    hf_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Uses 'token=' instead of deprecated argument 'use_auth_token'"
            )
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tie_word_embeddings: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to tie input and output embeddings for training invertible adapters."
        }
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_files: Optional[str] = field(
        default=None, metadata={"help": "The path to a specified training file."}
    )
    max_data_files: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of data files to download for hf datasets."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
        )
    train_file_ds: Optional[str] = field(
        default=None, metadata={"help": "Underlying dataset if input training data."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    multiling_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train a multilinugal adapter or not. Requires 'languages' and 'language_ratios' argument."}
    )
    languages: Optional[List[str]] = field(
        default_factory=lambda: list(),
        metadata={"help": "List of lnaguages (in ISO code) to be used for training. "}
    )
    language_ratios: Optional[List[str]] = field(
        default_factory=lambda: list(),
        metadata={"help": "An optioanl list containing the ratios of languages. Only required if 'multiling_adapter' argument is set."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=False, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    save_preprocessed_data: bool = field(
        default=False, metadata={"help": "Whether to save the preprocessed data files to disk or not."}
    )
    remove_empty_lines: bool = field(
        default=False, metadata={"help": "Whether to remove empty lines when loading a TXT file."}
    )
    sampler: Optional[str] = field(
        default="seq",
        metadata={
            "help": "The sampler to use for the DataLoaders"
        }
    )
    data_collator: str = field(
        default=None,
        metadata={
            "help":"What collate function to use. Currently only supports 'LM' as value."
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                if os.path.isfile(self.train_file):
                    extension = self.train_file.split(".")[-1]
                    assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
                if os.path.isdir(self.train_file):
                    files = [file_name for file_name in os.listdir(self.train_file)]
                    for file_name in files:
                        extension = file_name.split(".")[-1]
                        assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."

            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class AdditionalAdapterArguments:

    prompt_tuning: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use prompt tuning."}
    )
    inv_adapter: Optional[str] = field(
        default=None, metadata={"help": "add invertible adapter modules after the model embedding layer. Currently, this can be either nice or glow."}      
    )
    prompt_tuning_init_text: Optional[bool] = field(
        default=False, metadata={"help": "The initial prompt text to use for prompt tuning."}
    )
    task_type: Optional[str] = field(
        default=None, metadata={"help": "The type of task for which prompt tuning is being performed."}
    )
    num_virtual_tokens: Optional[int] = field(
        default=None, metadata={"help": "The number of virtual tokens to use for prompt tuning."}
    )
    prompt_tuning_init: Optional[str] = field(
        default="RANDOM", metadata={"help": "The initialization method for prompt tuning."}
    )
    peft_type: Optional[str] = field(
        default=None, metadata={"help": "The type of prompt tuning to use."}
    )
    task_prompt: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a task prompt for prompt tuning."}
    )
    fusion: Optional[str] = field(
        default="none", metadata={"help": "The fusion method to use for prompt tuning."}
    )
    language: Optional[str] = field(
        default="english", metadata={"help": "The language to use."}
    )
    partial_prompt_tuning_init_text: Optional[str] = field(
        default=None, metadata={"help": "The initial prompt text to use for prompt tuning."}
    )
    partial_embedding: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use partial embedding for prompt tuning."}
    )
    fixed_size: Optional[int] = field(
        default=0, metadata={"help": "The fixed size to use for prompt tuning."}
    )
    adapter_inference: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to evaluate a pre-trained language adapter at inference."
            )
        }
    )
    pretrained_adapter_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name/path to the pre-trained language adapter to evaluate."
            )
        }
    )
    reduction_factor: Optional[int] = field(
        default=None,
        metadata={
            "help":"Reduction factor of bottleneck adapter."
        }
    )
    inv_adapter_reduction_factor: Optional[int] = field(
        default=None,
        metadata={
            "help":"Reduction factor of invertible adapter."
        }
    )
    adapter_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Dropout probability for adapter layers."
        }
    )
    selfattn_lora: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whehter to add LoRA matrices to self-attention layers."
        }
    )
    intermediate_lora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whehter to add LoRA matrices to intermediate layers."
        }
    )
    output_lora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whehter to add LoRA matrices to output layers."
        }
    )
    attn_matrices: Optional[List[str]] = field(
        default_factory=lambda: ["q", "v"] if "adapters" in active_env else lambda: ["q_proj", "v_proj"],
        metadata={
            "help": "Which matrices are used as attention matrices."
        }
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": "alpha value for LoRA matrices."
        }
    )
    lora_rank: Optional[int] = field(
        default=None,
        metadata={
            "help": "Rank for LoRA matrices."
        }
    )
    adapter_drop_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "Specifies the ratio of hidden layers where no adapter layers should be added. Layers are dropped starting from the last layer."
        }
    )
    adapter_layers_ranges: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "A range specifying layers which adapter layers should be trained for."}
    )
    layers_ranges: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "List of layer ranges"}
    )


def load_text_files(train_file: str,
                    validation_split_percentage: int,
                    cache_dir: str,
                    preprocessing_num_workers: int,
                    keep_linebreaks: bool,
                    validation_file: str = None,
                    ):
    data_files = {}
    dataset_args = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    extension = (
        train_file.split(".")[-1]
        if train_file is not None
        else validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = keep_linebreaks
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
        num_proc=preprocessing_num_workers,
        **dataset_args,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{validation_split_percentage}%]",
            cache_dir=cache_dir,
            num_proc=preprocessing_num_workers,
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{validation_split_percentage}%:]",
            cache_dir=cache_dir,
            num_proc=preprocessing_num_workers,
            **dataset_args,
        )

    return raw_datasets

def parse_tuple(input_str: str) -> Tuple[int, int]:
    return ast.literal_eval(input_str)

def parse_tuple_list(input_list: List[str]) -> List[Tuple[int, int]]:
    return [parse_tuple(item) for item in input_list]

def cleanup():
    torch.distributed.destroy_process_group()

def main():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    #device = torch.device(f'cuda:{local_rank}')
    #torch.cuda.set_device(device)
    #torch.distributed.init_process_group('nccl')

    if "adapters" in active_env:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments, AdditionalAdapterArguments))

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args, adapter_args, add_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args, adapter_args, add_args = parser.parse_args_into_dataclasses()
    
    else:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalAdapterArguments))

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args, add_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args, add_args = parser.parse_args_into_dataclasses()       

    # set the wandb project where this run will be logged
    os.environ["WANDB_NAME"]=training_args.run_name
    print(f"WANDB Project Name: {os.getenv('WANDB_PROJECT', None)}")

     # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

    # JS: set env variable for HF token
    os.environ['HF_TOKEN'] = model_args.hf_token if isinstance(model_args.hf_token, str) else None 

    # define number of layers to train adapters for if needed
    if add_args.adapter_layers_ranges:
        #add_args.adapter_layers_ranges = parse_tuple_list(add_args.adapter_layers_ranges)
        if local_rank == 0:
            print("Provided adapter layer ranges:", add_args.adapter_layers_ranges)
            print(type(add_args.adapter_layers_ranges))
    
    if add_args.layers_ranges:
        assert len(add_args.layers_ranges) % 2 == 0, "Indicated ranges must be divisible by two to indicate start and end of range."
        add_args.layers_ranges = [int(l) for l in add_args.layers_ranges]
        list_ranges = [add_args.layers_ranges[i:i+2] for i in range(0, len(add_args.layers_ranges), 2)]
        add_args.layers_ranges = {idx for r in list_ranges for idx in range(r[0], r[1]+1)}
        print(f"Layer ranges new: {add_args.layers_ranges}")

    # languages
    LANGUAGE_MAPPING = get_language_mapping()
    CODE_2_LANG = code_2_lang()
    if local_rank == 0:
        print(f"Languages: {data_args.languages}")
        print(f"Language ratios: {data_args.language_ratios}")
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_token,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, tie_word_embeddings=model_args.tie_word_embeddings, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.hf_token,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    '''if "peft" in active_env:
        if add_args.peft_type == "PROMPT_TUNING":
            add_args.num_virtual_tokens = add_args.num_virtual_tokens if add_args.num_virtual_tokens else len(tokenizer(add_args.prompt_tuning_init_text)["input_ids"])
        else:
            add_args.num_virtual_tokens = None'''
            
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # todo: add multilingual support
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        assert isinstance(data_args.max_data_files, int), "HF datasets requires specified number of data files to download."

        data_files = get_hf_dataset_files(data_args.dataset_name,
                                          data_args.languages[0],
                                          data_args.max_data_files,
                                          )

        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.languages[0],
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            token=model_args.hf_token,
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.languages[0],
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                num_proc=data_args.preprocessing_num_workers,
                token=model_args.hf_token,
            )

            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.languages[0],
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                num_proc=data_args.preprocessing_num_workers,
                token=model_args.hf_token,
            )

        assert isinstance(data_args.languages[0], str)
        raw_datasets = {data_args.languages[0]: raw_datasets}
        if local_rank == 0:
            print(raw_datasets)

    elif data_args.train_file and data_args.train_file.endswith(".txt"):
        # for monolingual adapter
        print(f"Training monolingual adapter for {data_args.languages[0]}")
        raw_datasets = load_text_files(data_args.train_file,
                                       data_args.validation_split_percentage,
                                       model_args.cache_dir,
                                       data_args.preprocessing_num_workers,
                                       data_args.keep_linebreaks,
                                       data_args.validation_file,
                                       )
        assert isinstance(data_args.languages[0], str)
        raw_datasets = {data_args.languages[0]: raw_datasets}
        
    elif data_args.train_file and data_args.multiling_adapter:
        # for multilingual adapter
        assert os.path.isdir(data_args.train_file)
        print(data_args.languages)

        if len(data_args.languages) == 1:
            string_langs = data_args.languages[0]
            data_args.languages = [lang for lang in string_langs.split()]

        files_by_language = {lang: f"{data_args.train_file}/{data_args.train_file_ds}_{lang}_{data_args.max_train_samples}.txt" for lang in data_args.languages}
        print(files_by_language)
        #files_by_language = {file_path.split('.')[0]: os.path.join(data_args.train_file, file_path) for file_path in os.listdir(data_args.train_file)}

        raw_datasets = { language: load_text_files(file_path,
                                      data_args.validation_split_percentage,
                                      model_args.cache_dir,
                                      data_args.preprocessing_num_workers,
                                      data_args.keep_linebreaks,
                                      data_args.validation_file,
                                      ) for language, file_path in files_by_language.items()}

        if local_rank == 0:
            print(raw_datasets)

    else:
        raise ValueError("Provide a dataset accessible via load_dataset, a path to a .txt file or a directory containing .txt files.")


    # tokenization
    if training_args.do_train:
        column_names = {language: ds["train"].column_names for language, ds in raw_datasets.items()}
    else:
        column_names = {language: ds["train"].column_names for language, ds in raw_datasets.items()}
    
    print(f"Column names: {column_names}")
    
    first_lang_cols = next(iter(column_names.values()))
    text_column_name = "text" if "text" in first_lang_cols else first_lang_cols[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = {language: ds.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names[language],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        ) for language, ds in raw_datasets.items()}
    
    if local_rank == 0:
        print(tokenized_datasets)

    # packing
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = {language: ds.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        ) for language, ds in tokenized_datasets.items()}
    
    if local_rank == 0:
        print(lm_datasets)
    
    if data_args.remove_empty_lines:
        def remove_empty_lines(example):
            return example['text'].strip() != ''

        raw_datasets = raw_datasets.filter(remove_empty_lines, load_from_cache_file=not data_args.overwrite_cache)
    
    # concatenate datasets
    if data_args.multiling_adapter:
        assert isinstance(lm_datasets, dict)
        assert isinstance(data_args.language_ratios, list)

        train_language_dict = {lang: ds['train'] for lang, ds in lm_datasets.items()}
        eval_language_dict = {lang: ds['validation'] for lang, ds in lm_datasets.items()}

        if len(data_args.language_ratios) == 1:
            data_args.language_ratios = data_args.language_ratios[0]
        
        data_args.language_ratios = [float(ratio) for ratio in data_args.language_ratios]
        #print(f"Language ratios floats: {data_args.language_ratios}")
        
        assert sum(data_args.language_ratios) == 1, "Language ratios must sum up to 1."

        language_ratios = {lang: float(ratio) for lang, ratio in zip(list(lm_datasets.keys()), data_args.language_ratios)}
        main_process_print(language_ratios, local_rank)
        
        # alternative logic: determine indices for each language in the concatenated dataset
        def concatenate_multiling(ds_dict: Dict[str, datasets.Dataset], language_ratios: dict, batch_size: int):
            langs = list(ds_dict.keys())
            
            # determine number of samples per step per language by converting ratios of languages to corresponding int
            ratio_to_ints = utils.floats_to_ints(list(language_ratios.values()))
            assert len(ratio_to_ints) == len(list(language_ratios.keys()))
            num_samples_per_step = [ratio * batch_size for ratio in ratio_to_ints]

            main_process_print(f"Num samples per step: {num_samples_per_step}", local_rank)

            # determine total number of steps per language ds
            num_steps_per_lang = [len(ds_lang) // num_samples for ds_lang, num_samples in zip(ds_dict.values(), num_samples_per_step)]
            
            # choose smallest dataset as reference to avoid index errors; number of steps is balanced but the step size varies across languages 
            if len(set(num_steps_per_lang)) > 1:
                num_steps_per_lang = min(num_steps_per_lang)

            # check whether ds.select required
            dsets = [ds.select(range(num_steps_per_lang*num_samples)) for ds, num_samples in zip(ds_dict.values(), num_samples_per_step)]

            main_process_print(dsets, local_rank)
            
            # initialize an empty array having the size of the final concatenated ds
            total_length = sum(len(ds) for ds in dsets)
            concatenated_ds = np.empty(total_length, dtype=object)
            
            # determine offsets for the languages
            offsets = np.cumsum([0] + num_samples_per_step[:-1])

            # determine step size, i.e. how many samples are contained in one cycle (one step for each language)
            cycle_size = sum(num_samples_per_step)

            # starting from the respective offsets, determine indices of each language in the concatenated ds based on the step size and the number of samples per step per language
            indices_langs = [] 
            for i, offset in enumerate(offsets):
                indices_per_lang = []
                for j in range(offset, total_length, cycle_size):
                    indices_per_lang.extend(list(range(j, j + num_samples_per_step[i])))

                indices_langs.append(indices_per_lang)
                assert len(indices_per_lang) == len(dsets[i])
                
                # check whether needs to be converted to list 
                #lang_data = np.array([sample for sample in dsets[i]])
                #concatenated_ds[indices_per_lang] = lang_data

                #print("Language added to dataset.")

            idx_col = "target_idx"
            for i in range(len(dsets)):
                data = dsets[i]
                indices = indices_langs[i]

                print_idx = max(num_samples_per_step)
                main_process_print(f"language: {langs[i]} | first {print_idx} indices: {indices[:print_idx]}", local_rank)

                data = data.add_column(idx_col, indices)
                dsets[i] = data
            
            ds_concat = concatenate_datasets(dsets)
            ds_concat = ds_concat.sort(idx_col, load_from_cache_file=False)
            ds_concat = ds_concat.remove_columns(idx_col)

            print("Multilingual dataset created.")
            main_process_print(ds_concat, local_rank)

            return ds_concat
        
        with training_args.main_process_first(desc="Concatenating multilingual datasets.."):
            lm_datasets["train"] = concatenate_multiling(train_language_dict, language_ratios, training_args.per_device_train_batch_size)
            lm_datasets["validation"] = concatenate_multiling(eval_language_dict, language_ratios, training_args.per_device_eval_batch_size)
    
    else:
        lm_datasets = lm_datasets[data_args.languages[0]]

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", experiment_id=training_args.run_name)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            if isinstance(add_args.num_virtual_tokens, int):
                preds = preds[:, add_args.num_virtual_tokens:-1].reshape(-1)
            else:
                preds = preds[:, :-1].reshape(-1)
            
            accuracy = metric.compute(predictions=preds, references=labels)

            return {
                "accuracy": accuracy['accuracy']
            }

    if local_rank == 0:
        print(lm_datasets)

    if training_args.do_train:
        if model_args.model_name_or_path:
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.hf_token,
                torch_dtype=torch_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

        # Convert the model into an adapter model
        if "adapters" in active_env:
            adapters.init(model)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        # Setup adapters                            
        assert "/" in training_args.output_dir
        adapter_name = f"{training_args.output_dir.split('/')[-1]}" #_{data_args.dataset_name.split('/')[-1]}_{data_args.language}"

        if "." in adapter_name:
            adapter_name = adapter_name.replace(".", "")

        # setup adapter configurations for adapters libraray
        if "adapters" in active_env:
            if add_args.layers_ranges:
                layers_total = set(range(config.num_hidden_layers))
                layers_to_leave_out = list(sorted(layers_total - add_args.layers_ranges))

                if local_rank == 0:
                    print(f"\nTraining adapters for layers {add_args.layers_ranges}.\n")

            if adapter_args.adapter_config == "seq_bn_inv":
                adapter_config_kwargs = {
                    "inv_adapter_reduction_factor": add_args.inv_adapter_reduction_factor,
                    "inv_adapter": add_args.inv_adapter,
                    "reduction_factor": add_args.reduction_factor, 
                }

            elif adapter_args.adapter_config == "prompt_tuning":
                if not add_args.num_virtual_tokens and add_args.prompt_tuning_init == "from_string":
                    assert isinstance(add_args.prompt_tuning_init_text, str)
                    add_args.num_virtual_tokens = len(tokenizer(add_args.prompt_tuning_init_text)['input_ids'])

                adapter_config_kwargs = {
                    "prompt_length": add_args.num_virtual_tokens,
                    "prompt_init": add_args.prompt_tuning_init,
                    "prompt_init_text": add_args.prompt_tuning_init_text,
                }

                print(adapter_config_kwargs)
            
            elif adapter_args.adapter_config == "lora":
                adapter_config_kwargs = {
                    "selfattn_lora": add_args.selfattn_lora,
                    "intermediate_lora": add_args.intermediate_lora,
                    "output_lora": add_args.output_lora,
                    "attn_matrices": add_args.attn_matrices,
                    "alpha": add_args.lora_alpha,
                    "r": add_args.lora_rank,
                    "dropout": add_args.adapter_dropout,
                }

            # base: bn adapter
            else:
                adapter_config_kwargs = {
                    "dropout": add_args.adapter_dropout,
                    "reduction_factor": add_args.reduction_factor,
                    "leave_out": layers_to_leave_out if add_args.layers_ranges else [],
                }
                
            setup_adapter_training(model, adapter_args, adapter_name or "clm", adapter_config_kwargs=adapter_config_kwargs)
            trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
        
            if local_rank == 0:
                num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Trainable parameters: {num_trainable_params}")
                print(model)
                print(model.adapter_summary())
        
        # setup adapter configuration for PEFT library
        if "peft" in active_env:
            if isinstance(add_args.adapter_drop_ratio, float):
                num_layers = config.num_hidden_layers
                adapter_layers = list(range(num_layers - int(add_args.adapter_drop_ratio * num_layers)))
                print(f"Note: Drop adapters for layers {adapter_layers[-1] + 1} to {num_layers}.")

            peft_config_kwargs = {
                "base_model_name_or_path": model_args.model_name_or_path,
                "peft_type": add_args.peft_type,
                "task_type": add_args.task_type,
            }

            if add_args.peft_type == "LORA":
                peft_type_config_kwargs = {
                    "init_lora_weights": True,
                    "lora_alpha": add_args.lora_alpha,
                    "lora_dropout": add_args.adapter_dropout,
                    "r": add_args.lora_rank,
                    "target_modules": add_args.attn_matrices,
                    "layers_to_transform": adapter_layers if isinstance(add_args.adapter_drop_ratio, float) else None,
                }

            if add_args.peft_type == "PROMPT_TUNING":
                # todo: enable more flexible init text + multiling adapter
                if add_args.prompt_tuning_init == PromptTuningInit.TEXT and add_args.prompt_tuning_init_text:
                    add_args.prompt_tuning_init_text = f"Generate the output in {CODE_2_LANG[data_args.languages[0]]}:"

                
                peft_type_config_kwargs = {
                    "prompt_tuning_init": add_args.prompt_tuning_init,
                    "prompt_tuning_init_text": add_args.prompt_tuning_init_text,
                    "num_virtual_tokens": add_args.num_virtual_tokens,
                    "tokenizer_name_or_path": model_args.model_name_or_path,
                }

            # merge generic and peft-type-specific kwargs
            peft_config_kwargs = peft_config_kwargs | peft_type_config_kwargs
            peft_config = get_peft_config(peft_config_kwargs)
            print(peft_config)                            
            assert "/" in training_args.output_dir
            adapter_name = f"{training_args.output_dir.split('/')[-1]}"

            model = get_peft_model(model, peft_config, adapter_name)
            trainer_class = Trainer 

            print(model)
            print(model.print_trainable_parameters())

        assert data_args.sampler in ["seq", "sequential", "random"], "Sampler must be either 'sequential' or 'random'."
        data_args.sampler = SequentialSampler if data_args.sampler in ["seq", "sequential"] else RandomSampler 
        print(f"Sampler: {data_args.sampler}")

        def custom_get_train_dataloader(self) -> DataLoader:
            """
            Returns the training [`~torch.utils.data.DataLoader`].

            Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
            training if necessary) otherwise.

            Subclass and override this method if you want to inject some custom behavior.
            """
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                # modify sampler for mapstyle datasets
                dataloader_params["sampler"] = data_args.sampler(self.train_dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            dataloader = DataLoader(train_dataset, **dataloader_params)

            return self.accelerator.prepare(dataloader)

        def custom_get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
            """
            Returns the evaluation [`~torch.utils.data.DataLoader`].

            Subclass and override this method if you want to inject some custom behavior.

            Args:
                eval_dataset (`torch.utils.data.Dataset`, *optional*):
                    If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                    by the `model.forward()` method are automatically removed. It must implement `__len__`.
            """
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")

            # If we have persistent workers, don't do a fork bomb especially as eval datasets
            # don't change during training
            if hasattr(self, "_eval_dataloader") and self.args.dataloader_persistent_workers:
                return self.accelerator.prepare(self._eval_dataloader)
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            data_collator = self.data_collator

            if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = data_args.sampler(self.eval_dataset)
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            # accelerator.free_memory() will destroy the references, so
            # we need to store the non-prepared version
            eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
            if self.args.dataloader_persistent_workers:
                self._eval_dataloader = eval_dataloader

            return self.accelerator.prepare(eval_dataloader)

    
        # Initialize our Trainer
        trainer_class.get_train_dataloader = custom_get_train_dataloader
        trainer_class.get_eval_dataloader = custom_get_eval_dataloader

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) if data_args.data_collator == "LM" else default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
            callbacks=[PerplexityCallback()] if training_args.do_eval else None,
        )

        # Training
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model() 
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    if add_args.adapter_inference:
        model = AutoAdapterModel.from_pretrained(model_args.model_name_or_path)
        adapter_name = model.load_adapter(add_args.pretrained_adapter_name)
        model.set_active_adapters(adapter_name)

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_train and training_args.do_eval:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

    cleanup()

if __name__ == "__main__":
    main()