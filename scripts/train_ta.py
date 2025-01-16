# this script builds on top of https://github.com/adapter-hub/adapters/blob/main/examples/pytorch/language-modeling/run_clm.py

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
    from setup_adapter_training import setup_adapter_training_

elif active_env == "peft":
    from peft import (
    get_peft_model,
    get_peft_config, 
    PromptTuningConfig,
    PromptLearningConfig, 
    TaskType, 
    PromptTuningInit, 
    PeftModel, 
    PeftConfig,
    )

from types import MethodType
import wandb
import sys
import math
import logging
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List
import datasets
import torch
from datasets import load_dataset, DatasetDict
import evaluate
import torch.distributed
import torch.utils
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.utils.data
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint, seed_worker
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.utils.import_utils import is_datasets_available

from functools import partial
from collections import Counter
from importlib import reload
import aya_dataset as aya
import sib200_dataset as sib
import numpy as np
from utils import preprocess_batch, formatting, compute_metrics, string_to_list

logger = logging.getLogger(__name__)

def pretty_names(model, ds, use_la=False):
    if "Llama-2" in model:
        model = "llama-2"
    if "Llama-3" in model:
        model = "llama-3"
    if "Llama-3.1" in model:
        model = "llama-31"

    if "sib200" in ds:
        ds = "sib200"
    if "aya" in ds:
        ds = "aya"

    setting = "LA" if use_la else "noLA"

    return model, ds, setting

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
    # JS
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
    use_early_stopping: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to apply early stopping during training."
        }
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "When applying early stopping, stop training when the specified metric worsens for this number of evaluation calls."
        }
    )
    early_stopping_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "Denotes how much the specified metric must improve to satisfy early stopping conditions."
        }
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the wandb project to log in."
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
    mode: str = field(
        default="eval", 
        metadata={"help":"Whether to train, evaluate a task adapter or both."},
    )
    task: Optional[str] = field(
        default=None, metadata={"help": "The task to train the task adapter on"}
    )
    task_dataset: Optional[str] = field(
        default=None, metadata={"help": "A comma separated list of datasets the task adapter should be trained on"}
    )
    task_dataset_split: Optional[str] = field(
        default=None, metadata={"help": "Which split of the selected dataset to use."}
    )
    # JS: add 
    data_files: Optional[str] = field(
        default=None, metadata={"help": "An optional field specifying a data file of the dataset to use."}
    )
    ta_languages: str = field(
        default="english", metadata={"help": "Choose the source language for evaluating zero-shot cross-lingual transfer."}
    )
    lang_ratio: Optional[str] = field(
        default=1.0, metadata={"help": "Ratio of 'ta_languages'"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
        )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
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
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The fraction of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    instr_keys: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Whether to use specific instruction keys prepended to data samples.",
            "choices": ["chat", "engl_answer", "answer"]
            }
    )
    english_format: Optional[bool] = field(
        default=False,
        metadata={
            "help":"Whether to format the input data based on the English data (include linebreaks)."
        }
    )
    source_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help":"Whether to use English instructions for data samples in a target language"
        }
    )
    keep_en_instr: Optional[bool] = field(
        default=False,
        metadata={
            "help":"Whether to keep the instruction and markers in English while using labels in non-English language. Only for SIB-200"
        }
    )
    sampler: Optional[str] = field(
        default="random",
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

@dataclass
class AdditionalAdapterArguments:
    ta_name: str = field(
        default=None, metadata={"help": "Name of the task adapter to be trained."}
    )
    ta_path: Optional[str] = field(
        default=None, metadata={"help": "Name of directory for TA that is created as sub-directory of training_args.output_dir"}
    )
    la_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the language adapter, task adapter should be trained on top of."}
    )
    num_virtual_tokens: Optional[int] = field(
        default=None, metadata={"help": "The number of virtual tokens to use for prompt tuning."}
    )
    prompt_tuning_init: Optional[str] = field(
        default="random", metadata={"help": "The initialization method for prompt tuning."}
    )
    peft_type: Optional[str] = field(
        default=None, metadata={"help": "The type of PEFT to use."}
    )
    task_type: Optional[str] = field(
        default=None, metadata={"help": "The type of task to use, e.g. 'CAUSAL_LM'"}
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
    use_la_dummy_adapter: bool = field(
        default=False,
        metadata={
            "help": (
                "Will add an untrained dummy adapter as LA replacement.)."
            )
        },
    )
    fusion: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to train an adapter fusion module for LAs"
            )
        }
    )
    fuse_langs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Languages to fuse"
            )
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
        default_factory=lambda: ["q", "v"],
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
    merge_weights: bool = field(
        default=None,
        metadata={
            "help": "Whether to merge the LoRA weights of the LA with the base LLM prior to TA training."
        }
    )
    adapter_drop_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "Specifies the ratio of hidden layers where no adapter layers should be added. Layers are dropped starting from the last layer."
        }
    )
    reduction_factor: Optional[int] = field(
        default=None,
        metadata={
            "help":"Reduction factor of bottleneck adapter."
        }
    )

def cleanup():
    torch.distributed.destroy_process_group()

def main():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    torch.distributed.init_process_group('nccl')
    
    if "adapters" in active_env:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments, AdditionalAdapterArguments))
        model_args, data_args, training_args, adapter_args, add_args = parser.parse_args_into_dataclasses()
    else:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalAdapterArguments))
        model_args, data_args, training_args, add_args = parser.parse_args_into_dataclasses()      

    set_seed(training_args.seed)

    pretty_model, pretty_ds, pretty_setting = pretty_names(model_args.model_name_or_path, data_args.dataset_name, isinstance(add_args.la_name, str))

    # set the wandb project where this run will be logged
    os.environ["WANDB_WATCH"]="false"
    os.environ["WANDB_PROJECT"]=f"task_adapters_{pretty_model}_{pretty_ds}_abl"
    #print(f"WANDB Project Name: {os.getenv('WANDB_PROJECT', None)}")
    print(f"WANDB Project: {os.environ["WANDB_PROJECT"]}")

    adapter_name = f"{add_args.ta_name}_{training_args.seed}" 
    os.environ["WANDB_NAME"] = adapter_name
    training_args.run_name = adapter_name

    #login into and init wandb 
    wandb_api_key = os.getenv("WANDB_API_KEY", None)
    if wandb_api_key is not None:
        print('Logging into wandb')
        wandb.login(key=wandb_api_key)

    # save general output dir and overrride output dir to ensure checkpoints are saved in correct directory
    output_dir_ = training_args.output_dir
    training_args.output_dir = os.path.join(training_args.output_dir, add_args.ta_path, add_args.ta_name, adapter_name) if isinstance(add_args.ta_path, str) else os.path.join(training_args.output_dir, add_args.ta_name, adapter_name)
    print(f"Output dir: {training_args.output_dir}")

    data_args.ta_languages = string_to_list(data_args.ta_languages)
    data_args.lang_ratio = string_to_list(data_args.lang_ratio)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # JS: set env variable for HF token
    os.environ['HF_TOKEN'] = model_args.hf_token if isinstance(model_args.hf_token, str) else None

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
    #logger.info(f"Training/evaluation parameters {training_args}")

    # load data
    if data_args.dataset_name is not None:
        if "aya" in data_args.dataset_name:
            if data_args.task_dataset == "MLQA-en": data_args.task_dataset = "MLQA-en (T)"
            print(f"Dataset: {data_args.task_dataset}")

            raw_datasets = aya.make_dataset(
                                            dataset=data_args.dataset_name,
                                            mode=data_args.mode, 
                                            languages=data_args.ta_languages,
                                            lang_ratio=data_args.lang_ratio,
                                            cache_dir=model_args.cache_dir, 
                                            task=data_args.task,
                                            task_data=data_args.task_dataset,
                                            task_data_split=data_args.task_dataset_split,
                                            train_size=data_args.max_train_samples,
                                            eval_size=data_args.max_eval_samples,
                                            english_format=data_args.english_format,
                                            source_prompt=data_args.source_prompt,
                                            data_seed=training_args.data_seed,
                                            )
        
        elif "sib200" in data_args.dataset_name:
            raw_datasets = sib.make_dataset(
                                            data_args.dataset_name,
                                            data_args.mode,
                                            data_args.ta_languages,
                                            model_args.cache_dir,
                                            train_size=data_args.max_train_samples,
                                            eval_size=data_args.max_eval_samples,
                                            data_seed=training_args.data_seed,
                                            )

        else:
            raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_files=data_args.data_files,
            cache_dir=model_args.cache_dir,
        )
            
        if local_rank == 0:
            print(raw_datasets)

        if "validation" not in raw_datasets.keys() and training_args.do_eval:
            print("\nSplitting Dataset into Train, Validation, Test..\n")
            train_valid_split = raw_datasets['train'].train_test_split(test_size=data_args.validation_split_percentage, seed=training_args.data_seed)
            train_valid_split['validation'] = train_valid_split.pop('test')

            raw_datasets['train'] = train_valid_split['train']
            raw_datasets['validation'] = train_valid_split['validation']

            print(raw_datasets)

    # instantiate pre-trained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_token,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # JS: add token-arg
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
        )
    
    if "peft" in active_env:
        if add_args.peft_type == "PROMPT_TUNING":
            num_virtual_tokens = add_args.num_virtual_tokens if add_args.num_virtual_tokens else len(tokenizer(add_args.prompt_tuning_init_text)["input_ids"])
        else:
            num_virtual_tokens = None

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # JS add 'token'
        # consider using AutoAdapterModel Class
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
        raise ValueError(
            "You are instanitating a model from scratch. This is not supported by this script."
        )

    # Convert the model into an adapter model
    if "adapters" in active_env:
        adapters.init(model)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys() if k != 'text'}
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

    def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: datasets.DatasetDict):
        """
        Tokenizes dataset for fine-tuning

        :param tokenizer (AutoTokenizer): Model tokenizer
        :param max_length (int): Maximum number of tokens to emit from the tokenizer
        :param seed: Random seed for reproducibility
        :param dataset (str): Instruction dataset
        """

        # Add prompt to each sample
        print("Preprocessing dataset...")
        _formatting = partial(formatting, 
                              tokenizer=tokenizer, 
                              dataset=data_args.dataset_name, 
                              train=True, 
                              instr_keys=data_args.instr_keys, 
                              lang=data_args.ta_languages[0] if isinstance(data_args.ta_languages, List) else data_args.ta_languages, 
                              source=data_args.source_prompt,
                              keep_en_instr=data_args.keep_en_instr,
                              )
        dataset = dataset.map(_formatting, load_from_cache_file=False)

        print(dataset['train']['text'][0])

        # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
        )

        print(f"Dataset length before max length filtering: {len(dataset['train'])}")
        # Filter out samples that have "input_ids" exceeding "max_length"
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
        print(f"Dataset length after max length filtering: {len(dataset['train'])}")

        dataset = dataset.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        return dataset

    #with training_args.main_process_first(desc="Preprocessing.."):
    with training_args.main_process_first(desc="Preprocessing.."):
        lm_datasets = preprocess_dataset(tokenizer, max_length=1024, dataset=raw_datasets)
        print(lm_datasets)

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

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)
        
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Setup adapters                            
    #assert "/" in training_args.output_dir
    #adapter_name = training_args.output_dir.split('/')[-1]
    
    # change logic
    print(f"Adapter Name: {adapter_name}")

    # setup adapter config using adapters library
    if "adapters" in active_env:
        if adapter_args.adapter_config == "seq_bn_inv":
            adapter_config_kwargs = {
                "inv_adapter_reduction_factor": add_args.inv_adapter_reduction_factor, 
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

        else:
            adapter_config_kwargs = {
                "dropout": add_args.adapter_dropout,
                "reduction_factor": add_args.reduction_factor,
            }
    
        print(adapter_config_kwargs)

        if isinstance(add_args.la_name, str):
            la_path = f"{output_dir_}/{add_args.la_name}/{add_args.la_name}"
            adapter_args.load_lang_adapter = la_path
            adapter_args.lang_adapter_config = f"{la_path}/adapter_config.json"

        setup_adapter_training_(model, adapter_args, adapter_name or "ta", fusion=add_args.fusion, fuse_langs=add_args.fuse_langs, adapter_config_kwargs=adapter_config_kwargs)
        if local_rank == 0:
            print(model)
            print(model.adapter_summary())
        
        trainer_class = AdapterTrainer
    
    # setup adapter configuration for PEFT library
    if "peft" in active_env:
        if isinstance(add_args.adapter_drop_ratio, float):
            # todo: change logic; currently: drops first (ratio * num_layers) layers of model 
            num_layers = config.num_hidden_layers
            adapter_layers = list(range(int(add_args.adapter_drop_ratio * num_layers), num_layers))
            print(f"Note: Drop adapters for layers 0 to {adapter_layers[0] - 1}.")
        else:
            adapter_layers = None

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
                "layers_to_transform": adapter_layers,
            }

        if add_args.peft_type == "PROMPT_TUNING":
            if add_args.prompt_tuning_init == PromptTuningInit.TEXT:
                assert isinstance(add_args.prompt_tuning_init_text, str)
            
            peft_type_config_kwargs = {
                "prompt_tuning_init": add_args.prompt_tuning_init,
                "prompt_tuning_init_text": add_args.prompt_tuning_init_text,
                "num_virtual_tokens": num_virtual_tokens,
                "tokenizer_name_or_path": model_args.model_name_or_path,
            }

        # merge generic and peft-type-specific kwargs
        peft_config_kwargs = peft_config_kwargs | peft_type_config_kwargs
        ta_config = get_peft_config(peft_config_kwargs)
        print(ta_config)                            

        if isinstance(add_args.la_name, str):
            la_path = f"{output_dir_}/{add_args.la_name}/{add_args.la_name}"
            la_config = PeftConfig.from_pretrained(la_path)
            model = get_peft_model(model, la_config, add_args.la_name, mixed=True)
            for param in model.parameters():
                param.requires_grad = False    

            if add_args.merge_weights:
                model.merge_adapter()

            model.add_adapter(adapter_name, ta_config)
            model.set_adapter(adapter_name)
        
            print(f"Active adapters: {model.active_adapters}")

        else:
            model = get_peft_model(model, ta_config, adapter_name)

        if adapter_name != "default":
            from monkey_patches import custom_save_pretrained
            model.save_pretrained = custom_save_pretrained.__get__(model)
            print("\nPeftModel: 'save_pretrained' function modified.\n")

        trainer_class = Trainer 
        print(model)
        print(model.print_trainable_parameters())


    assert data_args.sampler in ["seq", "sequential", "random"], "Sampler must be either 'sequential' or 'random'."
    data_args.sampler = SequentialSampler if data_args.sampler in ["seq", "sequential"] else RandomSampler 

    # Initialize our Trainer
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

        print(f"Dataloader uses {dataloader_params["sampler"]} as sampler class.")

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

    if model_args.use_early_stopping:
        if (model_args.early_stopping_patience or model_args.early_stopping_threshold) is None:
            raise ValueError("Define patience and threshold in order to use early stopping.")
        
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience,    
            early_stopping_threshold=model_args.early_stopping_threshold
        )

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) if data_args.data_collator == "LM" else default_data_collator,
        callbacks=[early_stopping_callback] if model_args.use_early_stopping else [],
#        compute_metrics=compute_metrics if training_args.do_eval else None,
#        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Initialize our Trainer
    trainer.get_train_dataloader = MethodType(custom_get_train_dataloader, trainer)
    trainer.get_eval_dataloader = MethodType(custom_get_eval_dataloader, trainer)

    trainer.get_train_dataloader()

    print(f"Sampler: {data_args.sampler}")  

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        #trainer.save_model(output_dir=f"{training_args.output_dir}/{add_args.ta_name}/{adapter_name}")  # Saves the tokenizer too for easy upload
        trainer.save_model(output_dir=f"{training_args.output_dir}")

        metrics = train_result.metrics

        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # modify output dir for metrics logging when training multiple seeds
        #if model_args.multi_seed:
            #training_args.output_dir = f"{training_args.output_dir}/{adapter_name}"
            #os.makedirs(training_args.output_dir, exist_ok=True)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

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
    
    wandb.finish()
    cleanup()

if __name__ == "__main__":
    main()