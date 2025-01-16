import os
active_env = os.getenv('CONDA_DEFAULT_ENV')
if "adapters" in active_env:
    from adapters import (
        AutoAdapterModel, 
        LlamaAdapterModel, 
        AdapterConfig, 
        AdapterFusionConfig,
    )
    from adapters.composition import Stack
    import adapters

elif active_env == "peft":
    from peft import (
    get_peft_model, 
    PromptTuningConfig, 
    TaskType, 
    PromptTuningInit, 
    PeftModel, 
    PeftConfig,
    )

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, default_data_collator
from throughput import SampleRequest, sample_requests
import argparse
import aya_dataset as aya
import sib200_dataset as sib
import utils
import torch
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
import pandas as pd
from typing import Union, Any, Optional
import time
from utils import get_language_mapping, get_most_dist_lang, get_latest_checkpoint


LANGUAGE_MAPPING = get_language_mapping()
MOST_DIST_LANGS = get_most_dist_lang()

def load_adapters_ta_model(
                            model: Any,
                            use_ta: bool,
                            use_la: bool,
                            adapter_path: str,
                            task_adapter: str,
                            seed: int,
                            use_ta_head: Optional[bool] = False,
                            latest_checkpoint: Optional[bool] = False,
                            ta_path_format: Optional[str] = None,
                            adapter_method: Optional[str] = None,
                            merge_weights: Optional[bool] = False,
                           ):

    ta_name = f"{task_adapter}_{seed}"

    # todo: modify for older single seed runs
    if ta_path_format is not None:
        assert isinstance(ta_path_format, str)
        if ta_path_format == "depr":
            task_adapter_path = f"{adapter_path}/{task_adapter}/{ta_name}"
        elif ta_path_format == "old": 
            task_adapter_path = f"{adapter_path}/{task_adapter}/{ta_name}/{ta_name}/{ta_name}"
        elif latest_checkpoint and "final" in ta_path_format:
            # selects latest model checkpoint
            adapter_path_ = os.path.join(adapter_path, ta_path_format, task_adapter, ta_name)
            checkpoint = get_latest_checkpoint(adapter_path_)
            task_adapter_path = os.path.join(checkpoint, ta_name)
        elif "final" in ta_path_format:
            adapter_path_ = os.path.join(adapter_path, ta_path_format)
            assert os.path.isdir(adapter_path_), f"Directory {adapter_path_} doesn't exit "
            task_adapter_path = f"{adapter_path_}/{task_adapter}/{ta_name}/{ta_name}"

    else:
        task_adapter_path = f"{adapter_path}/{task_adapter}/{ta_name}/{ta_name}"

    task_adapter_config = AdapterConfig.load(f"{task_adapter_path}/adapter_config.json")
    
    if use_ta and not use_la:
        model.load_adapter(task_adapter_path, config=task_adapter_config, with_head=use_ta_head)
        model.active_adapters = ta_name
        print(model.adapter_summary())
        #model.set_active_adapters(ta_name)

        if adapter_method == "lora" and merge_weights:
            model.merge_adapter(ta_name)

    return model, task_adapter_path, task_adapter_config, ta_name

def load_adapters_la_model(
        model: Any,
        adapter_path: str,
        lang_code: str,
        lang_adapter_prefix: str,
        lang_adapter_suffix: Optional[str] = None,
        use_ta: Optional[bool] = False,
        ta_name: Optional[str] = None,
        task_adapter_path: Optional[str] = None,
        task_adapter_config: Optional[str] = None,
        use_ta_head: Optional[bool] = False,
        use_la_head: Optional[bool] = False,
        adapter_method: Optional[str] = None,
    ):

    la_suffix = f"_{lang_adapter_suffix}" if isinstance(lang_adapter_suffix, str) else ""
    la_name = f"{lang_adapter_prefix}_{lang_code}{la_suffix}"
        
    la_path = f"{adapter_path}/{la_name}/{la_name}"
    la_config_path = f"{la_path}/adapter_config.json"

    lang_adapter_config = AdapterConfig.load(la_config_path)
    model.load_adapter(la_path, config=lang_adapter_config, with_head=use_la_head)

    if use_ta and adapter_method == "seq_bn":
        model.load_adapter(task_adapter_path, config=task_adapter_config, with_head=use_ta_head)
        model.active_adapters = Stack(la_name, ta_name)
    else:
        model.active_adapters = la_name
    
    return model, la_name

def load_peft_ta_model(
                    base_model: Any,
                    cache_dir: str,
                    use_la: bool,
                    adapter_path: str,
                    task_adapter: str,
                    seed: int,
                    latest_checkpoint: Optional[bool] = False,
                    ta_path_format: Optional[str] = None,
                    merge_weights: Optional[bool] = False,
    ):

    ta_name = f"{task_adapter}_{seed}"

    if ta_path_format is not None:
        assert isinstance(ta_path_format, str)
        # todo: make more elegant, decouple checkpoint from ta path
        if latest_checkpoint and "final" in ta_path_format:
            adapter_path_ = os.path.join(adapter_path, ta_path_format, task_adapter, ta_name)
            task_adapter_path = get_latest_checkpoint(adapter_path_)  

        elif "final" in ta_path_format:
            adapter_path_ = os.path.join(adapter_path, ta_path_format)
            assert os.path.isdir(adapter_path_), f"Directory {adapter_path_} doesn't exit "
            task_adapter_path = f"{adapter_path_}/{task_adapter}/{ta_name}" 
    else:
        task_adapter_path = f"{adapter_path}/{task_adapter}/{ta_name}/{ta_name}"

    if not use_la:
        model = PeftModel.from_pretrained(
                base_model,
                task_adapter_path,
                adapter_name=ta_name,
                token=os.environ["HF_TOKEN"],
                cache_dir=cache_dir,
                is_trainable=False,
            )
        
        if merge_weights:
            model.merge_adapter()
    
        print(model)
    
    return model, task_adapter_path, ta_name

