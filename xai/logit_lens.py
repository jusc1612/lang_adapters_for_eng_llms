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
    from load_adapter_model import load_adapters_ta_model

elif active_env == "peft":
    from peft import (
    get_peft_model, 
    PromptTuningConfig, 
    TaskType, 
    PromptTuningInit, 
    PeftModel, 
    PeftConfig,
    )
    from load_adapter_model import load_peft_ta_model, load_adapters_la_model

import torch
from torch.utils.data import DataLoader
from tuned_lens.nn.lenses import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, set_seed
import argparse
from tuned_lens.plotting import PredictionTrajectory
import sib200_dataset as sib
import aya_dataset as aya
import utils
from functools import partial
from tqdm import tqdm
import time
import numpy as np
from typing import Optional
import pandas as pd
from datasets import Dataset

MODEL_MAPPING = {
    "meta-llama/Llama-2-7b-hf": "llama-2",
    "meta-llama/Meta-Llama-3-8B": "llama-3",
    "meta-llama/Meta-Llama-3.1-8B": "llama-31",
}

PRETTY_MODEL_MAPPING = {
    "meta-llama/Llama-2-7b-hf": "Llama 2",
    "meta-llama/Meta-Llama-3-8B": "Llama 3",
    "meta-llama/Meta-Llama-3.1-8B": "Llama 3.1",
}

LANGUAGE_MAPPING = {"afrikaans": "af", 
                    "catalan": "ca", 
                    "danish": "da", 
                    "german": "de", 
                    "english": "en",  
                    "finnish": "fi", 
                    "galician": "gl", 
                    "hungarian": "hu", 
                    "icelandic": "is", 
                    "dutch": "nl", 
                    "portuguese": "pt", 
                    "spanish": "es", 
                    "swedish": "sv"}

def update_eval_file(row, file_path):
    new_row = pd.DataFrame([row])
    
    if os.path.exists(file_path):
        scores = pd.read_csv(file_path)
        updated_file = pd.concat([scores, new_row], ignore_index=True)
    else:
        updated_file = new_row

    updated_file.to_csv(file_path, index=False)

def get_predictions(model, tokenizer, data, max_new_tokens, temperature=0.6, top_p=0.9, seed=42, stop_strings=None, cuda_det=False):
    predictions = []

    # standard values: temperature=0.6, top_p=0.9
    do_sample = top_p < 1
    print(f"Temperature: {temperature} ||| Top-p: {top_p} ||| Sampling: {do_sample}")

    #total_tokens = 0
    start_time = time.perf_counter()
    lengths_outputs = []

    for batch in tqdm(data, desc="Generating..."):

        set_seed(seed)

        if cuda_det:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        outputs = model.generate(**batch, 
                                 max_new_tokens=max_new_tokens, 
                                 pad_token_id=tokenizer.eos_token_id, 
                                 temperature=temperature, 
                                 top_p=top_p, 
                                 do_sample=do_sample,
                                 tokenizer=tokenizer,
                                 stop_strings=stop_strings,
                                 )
        
        #total_tokens += outputs.shape[-1]

        outputs = [output[len(input):] for input, output in zip(batch['input_ids'], outputs)]
        
        # determine length of newly generated tokens
        lengths = [len(output) for output in outputs]
        lengths_outputs.extend(lengths)

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(preds)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return predictions, lengths_outputs, elapsed_time

def make_plot(model, 
              tokenizer, 
              lens, 
              text, 
              layer_stride, 
              statistic, 
              token_range, 
              source: str, 
              target: str, 
              setting, 
              max_length=None,
              top_k: Optional[int] = None,
              ):
    input_ids = tokenizer.encode(text, max_length=max_length)
    targets = input_ids[1:] + [tokenizer.eos_token_id]

    if len(input_ids) == 0:
        raise ValueError("Please enter some text.")
    
    if (token_range[0] == token_range[1]):
        raise ValueError("Please provide valid token range.")
    
    pred_traj = PredictionTrajectory.from_lens_and_model(
        lens=lens,
        model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        targets=targets,
    )
    
    pred_traj_ = pred_traj.slice_sequence(slice(*token_range))

    if isinstance(top_k, int):
        topk_tokens, topk_values = pred_traj._get_topk_tokens_and_values(top_k, 
                                                                         sort_by=pred_traj.log_probs, 
                                                                         values=pred_traj.probs
                                                                         )
    else:
        topk_tokens = topk_values = None

    # previously: statistic in title
    return getattr(pred_traj_, statistic)().stride(layer_stride).figure(
        title=f"{lens.__class__.__name__}: {PRETTY_MODEL_MAPPING[model.name_or_path]} | setup: {setting} | source: {source.capitalize()} | target: {target.capitalize()}",
    ), topk_tokens, topk_values


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Base model that has been used for task adapter training')
    parser.add_argument('--hf_token', type=str, required=True, help='Personal HF token to access gated models like Llama 2')
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory to load model and data from')
    parser.add_argument('--overwrite_cache', action="store_true", help="Whether to overwrite the cache")
    parser.add_argument('--input_text', type=str, required=False, default=None, help='Single input sample, logit lens should be used for')
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to evaluate the adapter model on")
    parser.add_argument('--layer_stride', type=int, required=True, help='The layer stride applied to the plot')
    parser.add_argument('--statistic', type=str, required=True, choices=['entropy', 'cross_entropy', 'forward_kl'], help='The summary statistic to plot')
    parser.add_argument('--token_range', type=int, nargs="+", required=False, default=None, help="The token range to plot")
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where to save the plot and the topk tokens')
    parser.add_argument('--eval_size', type=int, required=False, help='Size of subset to evaluate on')
    parser.add_argument('--n_shots', type=int, required=False, default=None, help="Modify the number of shots to evaluate")
    parser.add_argument('--max_length', type=int, required=False, default=None, help="Maximum length of tokenized sequence.")
    parser.add_argument('--no_instr', action="store_true", help="Whether to add no instructions to each sample but only at the very top.")
    parser.add_argument('--add_eng_instr', action="store_true", help="Whehter to add single English instruction on top of the prompt. Only when 'no_instr' is set.")
    parser.add_argument('--source_prompt', type=str, required=False, default=None, help="Whether to use the instructions in the source language")
    parser.add_argument('--format_prompt', action="store_true", help="Whether to format the prompt as used for English.")
    parser.add_argument('--instr_keys', type=str, default=None, help="Whether to use specific instruction keys prepended to the sample")
    parser.add_argument('--source_lang', type=str, required=False, default=None, help="Source language to evaluate")
    parser.add_argument('--languages', metavar='N', type=str, nargs='+', required=True, help="Target languages to evaluate")
    #parser.add_argument('--target_lang', type=str, required=True, help="Target languages to evaluate")
    parser.add_argument('--language_ratios', type=str, required=False, default="1.0", help="Language ratios of languages to be evaluated")
    parser.add_argument('--task_dataset', type=str, required=False, default=None, help='Dataset corresponding to chosen task')
    parser.add_argument('--task_data_split', type=str, required=False, default=None, help='Split of chosen dataset to use.')
    parser.add_argument('--use_init', action="store_true", help="Whether to use init-Method to load AdapterModel or do it manually.")
    parser.add_argument('--task', type=str, required=False, default=None, help='Task to be evaluated')
    parser.add_argument('--xai_method', type=str, required=False, default=None, help="For bash script only.")

    parser.add_argument('--input_target', type=str, required=False, default=None, help="Determine input sample based on a provided target value")
    parser.add_argument('--target_index', type=int, required=False, default=None, help="Index of sample to visualize.")
    parser.add_argument('--use_la', action="store_true", help="Whether to use a language adapter for evaluation.")
    parser.add_argument('--use_ta', action="store_true", help="Whether to use a task adapter for evaluation.")
    parser.add_argument('--adapter_method', type=str, required=False, default=None, help="What adapter method to use")
    parser.add_argument('--seeds', type=int, required=False, default=None, help="Seed of the task adapter.")
    parser.add_argument('--merge_weights', action="store_true", help="Whether to merge weights at inference. For LoRA only.")
    parser.add_argument('--adapter_path', type=str, required=False, default=None, help="Path to the language adapter.")
    parser.add_argument('--task_adapter', type=str, required=False, default=None, help="Path to the task adapter.")
    parser.add_argument('--lang_adapter_prefix', type=str, required=False, default=None, help="Distinct prefix to identify language adapter to be loaded.")
    parser.add_argument('--lang_adapter_suffix', type=str, required=False, default=None, help="Distinct suffix to identify language adapter to be loaded.")
    parser.add_argument('--max_new_tokens', type=int, required=False, default=50, help="Maximum number of new tokens to be generated")
    parser.add_argument('--temperature', type=float, required=False, default=0.6, help="Temperature to use for generating")
    parser.add_argument('--top_p', type=float, required=False, default=0.9, help="Top-p sampling applied for generating")
    parser.add_argument('--stop_strings', nargs="+", type=str, required=False, default=None, help="List of strings to act as stop strings when generating.")
    parser.add_argument('--generation_seed', type=int, default=42, help="Sets seed to reproduce generations in case sampling is enabled.")
    parser.add_argument('--use_tuned_lens', action="store_true", help="Uses Tuned Lens as implemented in Belrose et al. (2023) if set else Logit Lens")
    parser.add_argument('--generate_output', action="store_true", help="Generates and prints the output as model produces using .generate() method.")
    parser.add_argument('--lang_specific_indices', type=str, required=False, default=None, help="Path to file containing indices of samples with language-specific targets (for MLQA only)")
    parser.add_argument('--eval_prefix', type=str, default=None, required=True, help="Unique identifier for saved file")
    parser.add_argument('--latest_checkpoint', action="store_true", help="Whether to evaluate the latest checkpoint being saved.")
    parser.add_argument('--ta_path_format', type=str, required=False, default=None, help="Path format for TAs. 'depr' for old Llama 2 TAs, 'old' for non-current ones.")
    parser.add_argument('--inlang', action="store_true", help="Use in-language demos for ICL.")
    parser.add_argument('--diff_instr', action="store_true", help="Use different instruction (from MAPLE paper) than standard template")
    parser.add_argument('--add_period', action="store_true", help="Adds period as additional halt signal to samples.")
    parser.add_argument('--topk', type=int, required=False, default=None, help="Extracts topk predicted token and saves them to csv file")
    parser.add_argument("--inp_passage", type=str, required=False, default=None, help="Extract target sample based on this passage occurring in the input. Takes the first one if not unique")
    parser.add_argument("--save_plots", action="store_true", help="Whether to save the heatmaps of the logit lenses")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_name_or_path = args.model_name_or_path
    os.environ["HF_TOKEN"] = args.hf_token
    cache_dir = args.cache_dir
    overwrite_cache = args.overwrite_cache
    input_text = args.input_text
    layer_stride = args.layer_stride
    statistic = args.statistic
    token_range = args.token_range
    save_dir = args.save_dir
    dataset_name = args.dataset
    n_shots = args.n_shots
    max_length = args.max_length
    source_lang = args.source_lang
    target_langs = args.languages
    source_prompt = args.source_prompt
    format_prompt = args.format_prompt
    no_instr = args.no_instr
    add_eng_instr = args.add_eng_instr
    instr_keys = args.instr_keys
    eval_size = args.eval_size
    input_target = args.input_target
    target_index = args.target_index
    task_dataset = args.task_dataset

    use_la = args.use_la
    use_ta = args.use_ta
    adapter_method = args.adapter_method
    task_adapter = args.task_adapter
    seed = args.seeds
    merge_weights = args.merge_weights
    lang_adapter_prefix = args.lang_adapter_prefix
    lang_adapter_suffix = args.lang_adapter_suffix
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    stop_strings = args.stop_strings
    generation_seed = args.generation_seed
    use_tuned_lens = args.use_tuned_lens
    generate_output = args.generate_output
    eval_prefix = args.eval_prefix

    adapter_path = args.adapter_path
    adapter_method = args.adapter_method 
    task_adapter = args.task_adapter

    latest_checkpoint = args.latest_checkpoint
    ta_path_format = args.ta_path_format
    inlang = args.inlang
    diff_instr = args.diff_instr
    add_period = args.add_period
    no_instr = args.no_instr
    add_eng_instr = args.add_eng_instr
    lang_specific_indices = args.lang_specific_indices
    topk = args.topk
    save_plots = args.save_plots

    if isinstance(args.inp_passage, str): 
        inp_passage = " ".join(args.inp_passage.split("_"))
    else:
        inp_passage = None

    print(inp_passage)

    if active_env == "peft" and (use_ta or use_la):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,token=os.environ["HF_TOKEN"], cache_dir=cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=os.environ['HF_TOKEN'], padding_side='left')

    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,token=os.environ["HF_TOKEN"], cache_dir=cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=os.environ['HF_TOKEN'], padding_side='left')

    print(f"Source langs: {source_lang}")
    print(f"Target langs: {target_langs}")

    setting = "LA" if use_la else "noLA"

    for target_lang in target_langs:
        
        # create adapter model
        if "adapters" in active_env:      
            if use_ta:
                adapters.init(model)
                assert isinstance(seed, int), "Requires a seed for task adapter."
                model, task_adapter_path, task_adapter_config, ta_name = load_adapters_ta_model(
                    model,
                    use_ta=use_ta,
                    use_la=use_la,
                    adapter_path=adapter_path,
                    task_adapter=task_adapter,
                    seed=seed,
                    latest_checkpoint=latest_checkpoint,
                    ta_path_format=ta_path_format,
                    )

            if use_la:
                adapters.init(model)
                lang_code = LANGUAGE_MAPPING[target_lang]
                la_suffix = f"_{lang_adapter_suffix}" if isinstance(lang_adapter_suffix, str) else ""
                la_name = f"{lang_adapter_prefix}_{lang_code}{la_suffix}"
                    
                la_path = f"{adapter_path}/{la_name}/{la_name}"
                la_config_path = f"{la_path}/adapter_config.json"

                lang_adapter_config = AdapterConfig.load(la_config_path)
                model.load_adapter(la_path, config=lang_adapter_config, with_head=False)

                if use_ta and adapter_method == "seq_bn":
                    model.load_adapter(task_adapter_path, config=task_adapter_config, with_head=False)
                    model.active_adapters = Stack(la_name, ta_name)
                else:
                    model.active_adapters = la_name
                
                print(model.adapter_summary())
            
            print(model)
            model.to(device)

        # instantiate lens model
        lens = TunedLens.from_model_and_pretrained(model) if use_tuned_lens else LogitLens.from_model(model)
        lens = lens.to(device)
        
        #logit_lens = LogitLens.from_model(model)

        # create samples
        if "aya" in dataset_name:
            target_column = "targets"
            if task_dataset == "MLQA-en": task_dataset = "MLQA-en (T)"

            if isinstance(n_shots, int):
                if n_shots > 0:
                    n_shot_examples = aya.make_dataset(dataset_name,
                                    mode="eval", 
                                    languages=target_lang if inlang else source_lang,
                                    lang_ratio=1.0,
                                    cache_dir=cache_dir,
                                    overwrite_cache=overwrite_cache, 
                                    train_size=n_shots,
                                    english_format=format_prompt,
                                    source_prompt=source_prompt,
                                    n_shot=True,
                                    diff_instr=diff_instr,
                                    no_instr=no_instr,
                                    )
                    print(n_shot_examples)
                    n_shot_examples = n_shot_examples['train']
                
                instr = utils.get_mlqa_instr() if no_instr and add_eng_instr else ""
                
            dataset = aya.make_dataset(dataset=dataset_name,
                                        mode="eval", 
                                        languages=target_lang,
                                        lang_ratio=1.0,
                                        cache_dir=cache_dir,
                                        overwrite_cache=overwrite_cache, 
                                        eval_size=eval_size,
                                        english_format=format_prompt,
                                        source_prompt=source_prompt,
                                        n_shot=isinstance(n_shots, int),
                                        diff_instr=diff_instr,
                                        no_instr=no_instr,
                                        )

        elif "sib200" in dataset_name:
            file_data_name = "sib200"
            target_column = "category"
            
            if isinstance(n_shots, int):
                if n_shots > 0:
                    n_shot_examples = sib.make_dataset(dataset_name,
                                    "train", 
                                    target_lang if inlang else source_lang,
                                    cache_dir, 
                                    train_size=n_shots,
                                    )
                    print(n_shot_examples)
                    n_shot_examples = n_shot_examples['train']
                
                instr = utils.get_sib200_instr() if no_instr and add_eng_instr else ""

            dataset = sib.make_dataset(dataset_name,
                                    "eval",
                                    target_lang,
                                    cache_dir,
                                    eval_size=eval_size,
                                    )
        
        else:
            raise ValueError("This script currently only supports MLQA-en and SIB-200 as datasets. Please select one of these two.")

        # main dataset formatting
        _formatting = partial(utils.formatting, 
                            tokenizer=tokenizer, 
                            dataset=dataset_name, 
                            instr_keys=instr_keys, 
                            lang=target_lang, 
                            source=source_prompt, 
                            n_shot=isinstance(n_shots, int), 
                            no_instr=no_instr,
                            add_period=add_period,
                            )
        
        dataset = dataset.map(_formatting, load_from_cache_file=not overwrite_cache)

        if "sib200" in dataset_name:
            inputs = dataset['test']['text']
            targets = dataset['test']['category']

        # few shot examples formatting
        if isinstance(n_shots, int):
            if n_shots > 0:
                _formatting = partial(utils.formatting, 
                                    tokenizer=tokenizer, 
                                    dataset=dataset_name, 
                                    instr_keys=instr_keys, 
                                    source=source_prompt, 
                                    train=True, 
                                    n_shot=True, 
                                    no_instr=no_instr,
                                    add_period=add_period,
                                    )
                
                n_shot_examples = n_shot_examples.map(_formatting, load_from_cache_file=not overwrite_cache)
            
            assert isinstance(instr, str)
            n_shot_prompt = instr + "\n\n".join(n_shot_examples['text']) if n_shots > 0 else instr
            assert isinstance(n_shot_prompt, str)
        
            def add_prompt(example):
                example['text'] = n_shot_prompt + "\n\n" + example['text']
                return example

            dataset = dataset.map(add_prompt, load_from_cache_file=False)

        # determine target sample
        if isinstance(target_index, int):
            target_index_ = file_id = target_index

        elif isinstance(inp_passage, str):
            # take english test set as reference to extarct sample 
            dataset_ref = sib.make_dataset(dataset_name,
                                    "eval",
                                    "english",
                                    cache_dir,
                                    eval_size=eval_size,
                                    )
            
            print(dataset_ref['test'])
            print(dataset_ref['test'][0])
            
            '''dataset_ref['test'] = dataset_ref['test'].filter(
                lambda example, index: inp_passage in example['text'][index],
                with_indices=True,
            )'''

            indices = [i for i, row in enumerate(dataset_ref['test']) if inp_passage in row['text']]
            print(f"Indices: {indices}")
            dataset_ref['test'] = dataset_ref['test'].select(indices)

            print(dataset_ref['test'][0])

            target_index_ = indices[0]
            print(target_index_)
            
            label = "-".join(dataset['test'][target_column][target_index_].split())
            file_id = f"{label}_{target_index_}"

        elif isinstance(lang_specific_indices, str):
            if not os.path.exists(lang_specific_indices):
                raise ValueError("Specified file path does not exist")

            else:
                indices = np.load(lang_specific_indices)
                dataset['test'] = dataset['test'].select(indices)
                target_index_ = 0
                label = "-".join(dataset['test'][target_column][target_index_].split())
                file_id = f"{label}_{target_index_}"
                print(dataset['test'][target_column][:10])

        # extract input sample based on target value
        elif isinstance(input_target, str):
            if isinstance(target_index, int):
                raise ValueError("Use either input_target or target_index to specify sample, not both.")
            
            dataset['test'] = dataset['test'].filter(lambda example: example[target_column] == input_target)
            label = "-".join(input_target.split())
            target_index_ = 0
            file_id = f"{label}_{target_index_}"
            print(dataset['test'])
        
        else:
            target_index_ = 0
            label = "-".join(dataset['test'][target_column][target_index_].split())
            file_id = f"{label}_{target_index_}"
        
        # select the sample of the provided index if provided, else selct the first sample 
        input_text = dataset['test']['text'][target_index_] 
        print(input_text)

        # determine max token range
        token_max = len(tokenizer.encode(input_text))

        if token_range is None:
            lang_token_range = [token_max - 20, token_max]
        elif len(token_range) == 1:
            lang_token_range = [token_max - token_range[0], token_max]
        else:
            lang_token_range = token_range
        
        print(lang_token_range)

        # generate predictions
        if generate_output:
            inputs = tokenizer(input_text, max_length=max_length)
            ds = Dataset.from_dict({
                "input_ids": [inputs["input_ids"]],
                "attention_mask": [inputs["attention_mask"]]
            })
            print(ds)
            ds = ds.with_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)
            print(ds)
            dataloader = DataLoader(ds, collate_fn=default_data_collator, batch_size=1)
            raw_preds, lengths_outputs, elapsed_time = get_predictions(model, 
                                        tokenizer, 
                                        dataloader, 
                                        max_new_tokens=max_new_tokens, 
                                        temperature=temperature, 
                                        top_p=top_p,
                                        seed=generation_seed,
                                        stop_strings=stop_strings,
                                    )
            
            print("---- predictions ---")
            for i, pred in enumerate(raw_preds):
                print(f"({i+1}):\t{pred}\n")

        plot, topk_tokens, topk_values = make_plot(
                                                model,
                                                tokenizer,
                                                lens=lens,
                                                text=input_text,
                                                layer_stride=layer_stride,
                                                statistic=statistic,
                                                token_range=lang_token_range,
                                                source=source_lang,
                                                target=target_lang,
                                                setting=setting,
                                                max_length=max_length,
                                                top_k=topk,
                                                )
        
        
        setup = "la" if use_la else "nola"
        if isinstance(n_shots, int): eval_prefix_ = f"icl-{n_shots}-{eval_prefix}"
        
        # todo: add distinction between TA and ICL
        if "aya" in dataset_name:
            ds_name = "aya"
        elif "sib200" in dataset_name:
            ds_name = "sib200"
        else:
            raise ValueError("Currently, no other dataset is supported in this script.")
        
        run_dir = f"{eval_prefix_}_{ds_name}_{MODEL_MAPPING[base_model_name_or_path]}_{source_lang}"

        plot_dir_ = os.path.join(save_dir, run_dir, str(file_id))
        os.makedirs(plot_dir_, exist_ok=True)
            
        if save_plots:
            print("Saving figure..")
            plot_name = f"{eval_prefix_}_{MODEL_MAPPING[base_model_name_or_path]}_{target_index_}_{ds_name}_{setup}_{source_lang}-{target_lang}"

            plot_path = os.path.join(plot_dir_, plot_name)
            print(plot_path)
            plot.write_image(f"{plot_path}.png")

        if isinstance(topk, int):
            # (33, 1678, 10) (1678, 10)
            print(f"{10 * '='} {target_lang} {10 * '='}")
            print(topk_tokens.shape)
            print(topk_tokens[0].shape)
            print("\nTop-k Tokens\n")
            print(topk_tokens)

            topk_dir = os.path.join(plot_dir_, f"{file_id}_top-{topk}_tokens")
            os.makedirs(topk_dir, exist_ok=True)
        
            topk_fn = f"top-{topk}_{eval_prefix_}_{MODEL_MAPPING[base_model_name_or_path]}_{target_index_}_{ds_name}_{setup}_{source_lang}.csv"
            topk_fn_ = os.path.join(topk_dir, topk_fn)
            for i, layer in enumerate(topk_tokens):
                final_tokens = layer[-1]
                print(final_tokens)
                
                final_tokens = {str(j+1): token for j, token in enumerate(final_tokens)}
                print(final_tokens)

                layer_lang_row = {
                    "language": target_lang,
                    "layer": i+1,
                    **final_tokens
                }

                update_eval_file(layer_lang_row, topk_fn_)
        
        # delete adapters after each target language
        if use_la:
            #if merge_weights:
                #model.unmerge_adapter()
            model.delete_adapter(la_name)
            if use_ta:
                model.delete_adapter(ta_name)

if __name__ == "__main__":
    main()