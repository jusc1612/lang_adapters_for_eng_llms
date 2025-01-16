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
    PeftMixedModel,
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
from typing import Union
import time
from utils import get_language_mapping, get_most_dist_lang


LANGUAGE_MAPPING = get_language_mapping()
MOST_DIST_LANGS = get_most_dist_lang()

def get_latest_checkpoint(adapter_dir):

    checkpoints = [d for d in os.listdir(adapter_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(adapter_dir, d))]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {adapter_dir}")

    checkpoint_numbers = [int(d.split("-")[-1]) for d in checkpoints]
    
    latest_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"

    print(f"Latest checkpoint: {latest_checkpoint}")
    
    return os.path.join(adapter_dir, latest_checkpoint)

def check_path(path_):
    print(f"Path exists: {os.path.exists(path_)}")

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

def update_eval_file(row, file_path):
    new_row = pd.DataFrame([row])
    
    if os.path.exists(file_path):
        scores = pd.read_csv(file_path)
        updated_file = pd.concat([scores, new_row], ignore_index=True)
    else:
        updated_file = new_row

    updated_file.to_csv(file_path, index=False)

def get_la_model(model, lang_adapter, lang_adapter_config, use_la_head, ta_name, task_adapter_path, task_adapter_config, use_ta_head, use_ta, adapter_method, device, merge_weigths=False):
    la_name = lang_adapter.split('/')[-1]
    lang_adapter_config = AdapterConfig.load(lang_adapter_config)
    model.load_adapter(lang_adapter, config=lang_adapter_config, with_head=use_la_head)

    if use_ta and adapter_method == "seq_bn":
        model.load_adapter(task_adapter_path, config=task_adapter_config, with_head=use_ta_head)
        model.active_adapters = Stack(la_name, ta_name)
    else:
        model.active_adapters = la_name

    if adapter_method == "lora" and merge_weigths:
        model.merge_adapter(la_name)
        if use_ta:
            model.merge_adapter(ta_name)

    return model.to(device)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True, help='Base model that has been used for task adapter training')
    parser.add_argument('--hf_token', type=str, required=True, help='Personal HF token to access gated models like Llama 2')
    parser.add_argument('--task_adapter', type=str, required=False, default=None, help='Path to trained task adapter')
    #parser.add_argument('--task_adapter_config', type=str, required=False, default=False, help='Path to task adapter configuration')
    parser.add_argument('--adapter_path', type=str, required=False, default=None, help='Path to the directory that stores all trained adapters')
    #parser.add_argument('--lang_adapter_config', type=str, default=None, help="Path to language adapter configuration")
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory to load model and data from')
    parser.add_argument('--model_cache_dir', type=str, required=False, default=None, help='Separate cache dir for base model')
    parser.add_argument('--overwrite_cache', action="store_true", help="Whether to overwrite the cache")
    parser.add_argument('--source_lang', metavar='N', type=str, nargs='+', required=False, default=None, help="Source language of task adapter")
    parser.add_argument('--languages', nargs="+", required=False, default=None, help='Language to evaluate task adapter on')
    parser.add_argument('--language_ratios', type=str, required=False, default="1.0", help="Language ratios of languages to be evaluated")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to evaluate the adapter model on")
    parser.add_argument('--task', type=str, required=False, help='Task to be evaluated')
    parser.add_argument('--task_dataset', type=str, required=False, help='Dataset corresponding to chosen task')
    parser.add_argument('--task_data_split', type=str, required=False, help='Split of chosen dataset to use.')
    parser.add_argument('--eval_size', type=int, required=False, help='Size of subset to evaluate on')
    parser.add_argument('--max_length', type=int, required=False, default=None, help='Maximum length of input batch')
    parser.add_argument('--max_new_tokens', type=int, required=False, default=100, help='Maximum number of new tokens to be generated')
    parser.add_argument('--save_preds', action="store_true", required=False, help="Whether to store the generated predictions and targets")
    parser.add_argument('--use_init', action="store_true", help="Whether to use init-Method to load AdapterModel or do it manually.")
    parser.add_argument('--use_la_head', action="store_true", help="Whether to use the causal LM prediction head of the trained LA")
    parser.add_argument('--use_ta_head', action="store_true", help="Whether to use the causal LM prediction head of the trained TA")
    parser.add_argument('--use_la', action="store_true", help="Whether to activate target LA or not")
    parser.add_argument('--use_ta', action="store_true", help="Whether to activate source TA or not")
    parser.add_argument('--eval_all', action="store_true", help="Whether to evaluate all settings for all languages")
    parser.add_argument('--setting', type=str, required=False, default=None, help="The evaluation setting. Only required if 'eval_all' flag is set")
    parser.add_argument('--n_shots', type=int, required=False, default=None, help="Modify the number of shots to evaluate")
    parser.add_argument('--inlang', action="store_true", help="For ICL, whether to use in-language demonstrations instead of source language.")
    parser.add_argument('--fusion_adapter', type=str, required=False, default=None, help="Set if adapter fusion module should be evaluated")
    parser.add_argument('--fuse_langs', metavar='N', type=str, nargs='+', required=False, default=None, help="Which languages to fuse their LAs with each other")
    parser.add_argument('--format_prompt', action="store_true", help="Whether to format instructions in the target language")
    parser.add_argument('--source_prompt', type=str, required=False, default=None, help="Code of source language to be used for instructions instead of target language instructions")
    parser.add_argument('--instr_keys', type=str, default=None, help="Whether to use specific instruction keys prepended to the sample")
    parser.add_argument('--save_preproc_data', action="store_true", help="Whether to save the preprocessed dataset (formatted and tokenized)")
    parser.add_argument('--adapter_method', type=str, default="bn", choices=["seq_bn", "seq_bn_inv", "prompt_tuning", "lora", "pt"], help="Adapter method used for fine-tuning")
    parser.add_argument('--lang_adapter_prefix', type=str, required=False, default=None, help="The shared prefix of each LA's name")
    parser.add_argument('--lang_adapter_suffix', type=str, required=False, default=None, help="The shared suffix of each LA's name")
    parser.add_argument('--seeds', nargs="+", type=int, required=False, default=None, help="List of seeds for TA to be evaluated.")
    parser.add_argument('--eval_prefix', type=str, required=False, default=None, help="Evaluation specific prefix that is prepended to file name.")
    parser.add_argument('--merge_weights', action="store_true", help="Whether to merge LoRA weights with the pre-trained weight matrices.")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature parameter for generating output. Defaults to 1.0 meaning deterministic output.")
    parser.add_argument('--top_p', type=float, default=1.0, help="Top p sampling parameter for output generation.")
    parser.add_argument('--generation_seed', type=int, default=42, help="Sets seed to reproduce generations in case sampling is enabled.")
    parser.add_argument('--diff_instr', action="store_true", help="Whether to apply MAPLE instruction to few-shot examples for MLQA.")
    parser.add_argument('--no_instr', action="store_true", help="Whether to add no instructions to each sample but only at the very top.")
    parser.add_argument('--add_eng_instr', action="store_true", help="Whehter to add single English instruction on top of the prompt. Only when 'no_instr' is set.")
    parser.add_argument('--add_period', action="store_true", help="Whether to add a period to targets of training/in-context learning examples.")
    parser.add_argument('--stop_strings', metavar='N', nargs="+", type=str, required=False, default=None, help="List of strings to act as stop strings when generating.")
    parser.add_argument('--stop_at_first_upper', action="store_true", help="Whether to cut off generations at first upper-case letter occurred. Only for n-shot and SIB-200")
    parser.add_argument('--do_throughput', action="store_true", help="Whether to save throughput scores.")
    parser.add_argument('--ta_path_format', type=str, required=False, default=None, help="Path format for TAs. 'depr' for old Llama 2 TAs, 'old' for non-current ones.")
    parser.add_argument('--eval_dir', type=str, required=False, default=None, help="Evaluation directory path to save file in. Creates a dir in current working dir if not provided.")
    parser.add_argument('--multiling', action="store_true", help="Whether the LA to be evaluated is multilingual or not.")
    parser.add_argument('--latest_checkpoint', action="store_true", help="Whether to evaluate the latest checkpoint being saved.")
    parser.add_argument('--dist_la', action="store_true", help="Whether to use the language of the most distant language (based in lang2vec) to assess impact of language-specific LA.")
    parser.add_argument('--one_for_all', type=str, required=False, default=None, help="Language whose LA should be used for all target languages at inference.")
    parser.add_argument('--keep_en_instr', action="store_true", help="Whether to keep English instructions but labels in non-English")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["HF_TOKEN"] = args.hf_token
    base_model_name_or_path = args.model_name_or_path
    task_adapter = args.task_adapter
    #task_adapter_config = args.task_adapter_config
    adapter_path = args.adapter_path
    #lang_adapter_config = args.lang_adapter_config
    cache_dir = args.cache_dir
    model_cache_dir = args.model_cache_dir
    overwrite_cache = args.overwrite_cache
    languages = args.languages
    language_ratios = utils.string_to_list(args.language_ratios)
    source_lang = args.source_lang
    dataset_name = args.dataset
    task = args.task
    task_dataset = args.task_dataset
    task_data_split = args.task_data_split
    eval_size = args.eval_size
    max_length = args.max_length
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    generation_seed = args.generation_seed
    save_preds = args.save_preds
    use_init = args.use_init
    use_la_head = args.use_la_head
    use_ta_head = args.use_ta_head
    use_la = args.use_la
    use_ta = args.use_ta
    eval_all = args.eval_all
    setting = args.setting
    n_shots = args.n_shots
    inlang = args.inlang
    fusion_adapter = args.fusion_adapter
    source_prompt = args.source_prompt
    format_prompt = args.format_prompt
    instr_keys = args.instr_keys
    lang_adapter_prefix = args.lang_adapter_prefix
    lang_adapter_suffix = args.lang_adapter_suffix
    seeds = args.seeds
    eval_prefix = args.eval_prefix
    adapter_method = args.adapter_method
    merge_weights = args.merge_weights
    diff_instr = args.diff_instr
    no_instr = args.no_instr
    add_eng_instr = args.add_eng_instr
    add_period = args.add_period
    stop_strings = args.stop_strings
    stop_at_first_upper = args.stop_at_first_upper
    do_throughput = args.do_throughput
    ta_path_format = args.ta_path_format
    eval_dir = args.eval_dir
    multiling = args.multiling
    latest_checkpoint = args.latest_checkpoint
    dist_la = args.dist_la
    one_for_all = args.one_for_all
    keep_en_instr = args.keep_en_instr

    if multiling and not use_la:
        raise ValueError("'multiling' requires multilingual LA to be evaluated.")
    
    # set seed for output reproducibility 
    set_seed(generation_seed)

    print(f"Source language: {', '.join(source_lang) if len(source_lang) > 1 else source_lang[0]}")
    print(f"Stop strings: {stop_strings}")

    if (use_ta or use_la) and (fusion_adapter == None) and ("adapters" in active_env) :
        if isinstance(n_shots, int) and use_ta:
            raise ValueError("Combining few-shot evaluation and adapter training is not implemented in this script yet.")
        if 'llama' in base_model_name_or_path:
            if use_init:
                model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,token=os.environ["HF_TOKEN"], cache_dir=model_cache_dir if model_cache_dir else cache_dir).to(device)
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=os.environ['HF_TOKEN'], padding_side='left')
                adapters.init(model)
            else:
                model = AutoAdapterModel.from_pretrained(base_model_name_or_path,token=os.environ["HF_TOKEN"], cache_dir=cache_dir).to(device)
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=os.environ['HF_TOKEN'], padding_side='left')
        else:
            if use_init:
                model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, cache_dir=cache_dir).to(device)
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
                adapters.init(model)
            else:
                model = AutoAdapterModel.from_pretrained(base_model_name_or_path, cache_dir=cache_dir).to(device)
                tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    elif active_env == "peft" and (use_ta or use_la):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,token=os.environ["HF_TOKEN"], cache_dir=cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=os.environ['HF_TOKEN'], padding_side='left')

    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,token=os.environ["HF_TOKEN"], cache_dir=cache_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, token=os.environ['HF_TOKEN'], padding_side='left')

        print(model)
    

    seeds = [42] if seeds is None else seeds
    #print(f"Seeds: {', '.join(seeds) if len(seeds) > 1 else seeds[0]}")

    all_scores = pd.DataFrame({"lang": [], "seed": [], "em": [], "f1": []})

    for seed in seeds:
        
        if use_ta and "adapters" in active_env:
            #ta_name = f"{task_adapter.split('/')[-1]}_{seed}"
            # add seq for sequential LAs
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

            model.to(device)

        if "peft" in active_env and use_ta:
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

        for lang in languages:

            if "adapters" in active_env and use_la:
                if multiling:
                    lang_code = f"{'_'.join([LANGUAGE_MAPPING[l] for l in languages])}"
                elif dist_la:
                    if isinstance(one_for_all, str):
                        lang_code = LANGUAGE_MAPPING[one_for_all]
                    else:
                        lang_code = MOST_DIST_LANGS[LANGUAGE_MAPPING[lang]]
                else:
                    lang_code = LANGUAGE_MAPPING[lang]

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

                if adapter_method == "lora" and merge_weights:
                    model.merge_adapter(la_name)
                    if use_ta:
                        model.merge_adapter(ta_name)

                print(model)
                print(model.adapter_summary())
                model.to(device)

            if "peft" in active_env and use_la:
                if dist_la:
                    if isinstance(one_for_all, str):
                        la_name =  f"{lang_adapter_prefix}_{LANGUAGE_MAPPING[one_for_all]}_{lang_adapter_suffix}"
                    else:
                        la_name =  f"{lang_adapter_prefix}_{MOST_DIST_LANGS[LANGUAGE_MAPPING[lang]]}_{lang_adapter_suffix}"
                else:
                    la_name =  f"{lang_adapter_prefix}_{LANGUAGE_MAPPING[lang]}_{lang_adapter_suffix}"
                la_path = f"{adapter_path}/{la_name}/{la_name}"

                print(la_path)

                #peft_model_class = PeftMixedModel if use_la and use_ta else PeftModel
                peft_model_class = PeftModel
                model = peft_model_class.from_pretrained(
                    base_model,
                    la_path,
                    adapter_name=la_name,
                    token=os.environ["HF_TOKEN"],
                    cache_dir=cache_dir,
                    is_trainable=False,
                )

                print(f"Active adapters: {model.active_adapter}")
                
                if adapter_method == "lora" and merge_weights:
                    model.merge_adapter(la_name)

                model.set_adapter(la_name)
                print(f"Active adapters: {model.active_adapter}")

                if use_ta:
                    print("\nNow loading task adapter..\n")
                    model.load_adapter(task_adapter_path, adapter_name=ta_name)
                    model.set_adapter(ta_name)

                if adapter_method == "lora" and merge_weights:
                    #model.merge_adapter(adapter_names=[ta_name])
                    model.merge_adapter()

                #print(f"Active adapters: {model.active_adapters}")

                #model.to_device()
                print(model)

            if "aya" in dataset_name:
                file_data_name = "aya"
                target_column = "targets"
                if task_dataset == "MLQA-en": task_dataset = "MLQA-en (T)"
                
                if isinstance(n_shots, int):
                    if n_shots > 0:
                        n_shot_examples = aya.make_dataset(dataset_name,
                                        mode="eval", 
                                        languages=lang if inlang else source_lang,
                                        lang_ratio=language_ratios,
                                        cache_dir=cache_dir,
                                        overwrite_cache=overwrite_cache, 
                                        task=task,
                                        task_data=task_dataset,
                                        task_data_split=task_data_split,
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
                                            languages=lang,
                                            lang_ratio=language_ratios,
                                            cache_dir=cache_dir,
                                            overwrite_cache=overwrite_cache, 
                                            task=task,
                                            task_data=task_dataset,
                                            task_data_split=task_data_split,
                                            eval_size=eval_size,
                                            english_format=format_prompt,
                                            source_prompt=source_prompt,
                                            n_shot=isinstance(n_shots, int),
                                            diff_instr=diff_instr,
                                            no_instr=no_instr,
                                            )

                targets = dataset['test']['targets']
                inputs = dataset['test']['inputs']
            
            if "sib200" in dataset_name:
                file_data_name = "sib200"
                target_column = "category"
                if isinstance(n_shots, int):
                    if n_shots > 0:
                        n_shot_examples = sib.make_dataset(dataset_name,
                                        "train", 
                                        lang if inlang else source_lang,
                                        cache_dir, 
                                        train_size=n_shots,
                                        )
                        print(n_shot_examples)
                        n_shot_examples = n_shot_examples['train']

                    instr = utils.get_sib200_instr() if no_instr and add_eng_instr else ""

                dataset = sib.make_dataset(dataset_name,
                                        "eval",
                                        lang,
                                        cache_dir,
                                        eval_size=eval_size,
                                        )

            _formatting = partial(utils.formatting, 
                                  tokenizer=tokenizer, 
                                  dataset=dataset_name, 
                                  instr_keys=instr_keys, 
                                  lang=lang, 
                                  source=source_prompt, 
                                  n_shot=isinstance(n_shots, int), 
                                  no_instr=no_instr,
                                  add_period=add_period,
                                  keep_en_instr=keep_en_instr,
                                )
            
            #_formatting = partial(utils.formatting, tokenizer=tokenizer, dataset=dataset_name)
            dataset = dataset.map(_formatting, load_from_cache_file=not overwrite_cache)

            if "sib200" in dataset_name:
                inputs = dataset['test']['text']
                targets = dataset['test']['category']

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
                                        keep_en_instr=keep_en_instr,
                                        )
                    n_shot_examples = n_shot_examples.map(_formatting, load_from_cache_file=not overwrite_cache)
                
                assert isinstance(instr, str)
                n_shot_prompt = instr + "\n\n".join(n_shot_examples['text']) if n_shots > 0 else instr
                assert isinstance(n_shot_prompt, str)
                
                def add_prompt(example):
                    example['text'] = n_shot_prompt + "\n\n" + example['text'] if n_shots > 0 else n_shot_prompt + example['text']
                    return example

                dataset = dataset.map(add_prompt, load_from_cache_file=False)      

            print(dataset['test'][0]['text'])
            max_length = max_length if isinstance(max_length, int) else tokenizer.model_max_length
            print(f"Input max length: {max_length}")
            _preprocessing_function = partial(utils.preprocess_batch, max_length=max_length, tokenizer=tokenizer, column="text")
            dataset_tok = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=dataset['test'].column_names,
                load_from_cache_file=not overwrite_cache,
            )

            if isinstance(eval_size, int) and eval_size <= 20:
                inp_lenghts = [len(inp) for inp in dataset_tok['test']['input_ids']]
            print(len(dataset_tok['test']['input_ids'][0]))

            if do_throughput:
                _preprocessing_function = partial(utils.preprocess_batch, max_length=max_length, tokenizer=tokenizer, column=target_column)
                targets_tok = dataset.map(
                    _preprocessing_function,
                    batched=True,
                    remove_columns=dataset['test'].column_names,
                    load_from_cache_file=not overwrite_cache,
                )

                print(dataset_tok)
                print(targets_tok)

                throughput_ds = sample_requests(dataset_tok,
                                                targets_tok,
                                                max_length,
                                                )
                print(throughput_ds)
                
            # generate predictions
            dataset_tok = dataset_tok.with_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)
            dataloader = DataLoader(dataset_tok['test'], collate_fn=default_data_collator, batch_size=1)
            raw_preds, pred_lengths, elapsed_time = get_predictions(model, 
                                                                    tokenizer, 
                                                                    dataloader, 
                                                                    max_new_tokens=max_new_tokens, 
                                                                    temperature=temperature, 
                                                                    top_p=top_p,
                                                                    seed=generation_seed,
                                                                    stop_strings=stop_strings,
                                                                    )

            # delete la in case la setup is evaluated; rely on base model re-init for pt since delete_adapter() is not supported 
            if use_la and adapter_method != 'pt':
                if adapter_method == "lora" and merge_weights:
                    model.unmerge_adapter()
                model.delete_adapter(la_name)
                if use_ta:
                    model.delete_adapter(ta_name)

            if eval_size is not None and eval_size < 50:
                print("---------------")
                print(raw_preds)
                print("----------------")

            # evaluate
            if stop_strings is not None:
                def remove_stop_strings(preds, strings_to_remove):
                    result = []
                    for string in preds:
                        clean_pred = []
                        if "sib200" in dataset_name and stop_at_first_upper:
                            first_word = string.split()[0] if len(string.split()) > 0 else string
                            if len(first_word) > 0:
                                first_word = first_word[:1].lower() + first_word[1:]
                            def extract_until_capital(s):
                                result = []
                                for char in s:
                                    if char.isupper():
                                        break
                                    result.append(char)
                                return ''.join(result).strip()
                            
                            clean_pred.append(extract_until_capital(first_word))

                        else:
                            for word in string.split():
                                # prev: not in and second line append
                                if word in strings_to_remove:
                                    break
                                clean_pred.append(word)
                        
                        clean_pred = " ".join(clean_pred)
                        result.append(clean_pred)
                                
                    return result

                raw_preds_ = remove_stop_strings(raw_preds, stop_strings)
            else:
                raw_preds_ = raw_preds

            metrics = utils.compute_metrics(raw_preds_, targets)
            
            for metric, score in metrics.items():
                print(f"{metric}:\t\t{score}")

            scores = {"lang": lang, "seed": int(seed), "em": metrics["em"], "f1": metrics["f1"]}
            all_scores = pd.concat([all_scores, pd.DataFrame([scores])], ignore_index=True)

            preds = [utils.normalize_answer(pred) for pred in raw_preds_]
            targets_ = [utils.normalize_answer(target) for target in targets]
            
            if eval_size is not None and eval_size <= 50:
                print("Raw predictions")
                print(raw_preds)
                print("--------------------------------")
                print("Normalized predictions")
                print(preds)
                print("--------------------------------")
                print(targets_)

            if save_preds or eval_all or do_throughput:
                if "Llama-2" in base_model_name_or_path:
                    model_name = "llama-2"
                if "Llama-3" in base_model_name_or_path:
                    model_name = "llama-3"
                if "Llama-3.1" in base_model_name_or_path:
                    model_name = "llama-31"
                if use_la:
                    if "cc100" in la_name:
                        file_la_name = "cc100"

                if "sib200" in dataset_name:
                    file_ta_name = "sib200"
                if "aya" in dataset_name:
                    file_ta_name = "aya"

                #if not isinstance(setting, str):
                setting = "LA" if use_la else "noLA"

            if len(source_lang) > 1:
                source_lang_file = "_".join([LANGUAGE_MAPPING[sl] for sl in source_lang])
            else:
                source_lang_file = LANGUAGE_MAPPING[source_lang[0]]
            
            if len(languages) > 1:
                langs = "_".join(languages)
            else:
                langs = languages[0]

            # save model predictions
            if save_preds:
                eval_dir_preds = f"{eval_dir}_preds" if isinstance(eval_dir, str) else os.path.join(os.getcwd(), "model_outputs")
                print(eval_dir_preds)
                
                os.makedirs(eval_dir_preds, exist_ok=True)

                if not os.path.exists(eval_dir_preds):
                    raise OSError(f"Failed to create the directory: {eval_dir_preds}")

                eval_prefix_ = f"{eval_prefix}_" if isinstance(eval_prefix, str) else ""
                seed = seed if isinstance(seed, int) else "na"

                file_name = f"{eval_prefix_}{model_name}_{source_lang_file}_{LANGUAGE_MAPPING[lang]}_{file_data_name}_{seed}" 
                print(file_name)
                
                preds_to_save = {"inputs": inputs, "targets_raw": targets, "targets_norm": targets_, "preds_raw": raw_preds, "preds_norm": preds}
                if file_ta_name == "sib200": preds_to_save.update({"raw_preds_stops": raw_preds_})

                if use_la or use_ta:
                    pd.DataFrame(preds_to_save).to_csv(f"{eval_dir_preds}/{file_name}-{setting.lower()}.csv", index=False)
                elif isinstance(n_shots, int):
                    pd.DataFrame(preds_to_save).to_csv(f"{eval_dir_preds}/{file_name}_{n_shots}-shot_{setting.lower()}.csv", index=False)

            if eval_all and (eval_size is None):
                eval_dir = eval_dir if isinstance(eval_dir, str) else os.path.join(os.getcwd(), "eval_files")
                print(eval_dir)
                
                os.makedirs(eval_dir, exist_ok=True)

                if not os.path.exists(eval_dir):
                    raise OSError(f"Failed to create the directory: {eval_dir}")

                eval_prefix_ = f"{eval_prefix}_" if isinstance(eval_prefix, str) else ""

                file_name = f"{eval_prefix_}{model_name}_{source_lang_file}_{file_data_name}.csv"
                file_path = os.path.join(eval_dir, file_name)
                print(file_path)

                seed = seed if isinstance(seed, int) else "na" 

                setting_lang = {"setting": setting, "language": lang, "seed": seed}
                new_row = {**setting_lang, **metrics}
                update_eval_file(new_row, file_path)
            
            if do_throughput:
                #print(f"Throughput for {lang}: {elapsed_time} tokens/s")
                eval_dir = eval_dir if isinstance(eval_dir, str) else os.path.join(os.getcwd(), "throughput")
                os.makedirs(eval_dir, exist_ok=True)

                file_name_tp = f"throughput_{model_name}_{file_data_name}_{seed}.csv"
                file_path_tp = os.path.join(eval_dir, file_name_tp)
                
                # ({"model": [], "source": [], "target": [], "setting": [], "adapter": [], "throughput": []})
                adapter_method_ = f"{adapter_method}_merged" if adapter_method == "lora" and merge_weights else adapter_method

                # throughput scores
                total_num_tokens = sum(request.prompt_len + request.expected_output_len
                           for request in throughput_ds)
                total_expected_output_tokens = sum(request.expected_output_len
                              for request in throughput_ds)
                total_output_tokens = sum(pred_lengths)
    
                print(f"Throughput: {len(throughput_ds) / elapsed_time:.2f} requests/s, "
                    f"{total_num_tokens / elapsed_time:.2f} total tokens/s, "
                    f"{total_expected_output_tokens / elapsed_time:.2f} expected output tokens/s, "
                    f"{total_output_tokens / elapsed_time:.2f} output tokens/s")

                row_tp = {"model": model_name, 
                          "ds": file_data_name, 
                          "setting": setting, 
                          "source": source_lang_file, 
                          "target": lang, 
                          "adapter": adapter_method_, 
                          "elapsed_time": elapsed_time,
                          "num_samples": len(throughput_ds),
                          "total_num_tokens": total_num_tokens,
                          "samples_per_second": len(throughput_ds) / elapsed_time,
                          "tokens_per_second": total_num_tokens / elapsed_time,
                          "total_expected_output_tokens": total_expected_output_tokens / elapsed_time,
                          "output_tokens_per_second": total_output_tokens / elapsed_time,
                          }
                
                update_eval_file(row_tp, file_path_tp)
        
        # delete ta after each seed
        if use_ta and not use_la and adapter_method != 'pt':
            if "peft" in active_env and adapter_method == "lora" and merge_weights:
                model.unmerge_adapter()
            
            model.delete_adapter(ta_name) 
    

    print("\n### Scores Summary ###\n")
    print(all_scores)

if __name__ == "__main__":
    main()