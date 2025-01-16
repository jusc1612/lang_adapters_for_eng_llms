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
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import Any, List, Union, Optional
import argparse
import aya_dataset as aya
import sib200_dataset as sib
import utils
from functools import partial
import pickle
from typing import Optional

set_seed(42)

LANGUAGE_MAPPING = utils.get_language_mapping()

def get_la_model(model, lang_adapter, lang_adapter_config, use_la_head, ta_name, device):
    la_name = lang_adapter.split('/')[-1]
    lang_adapter_config = AdapterConfig.load(lang_adapter_config)
    model.load_adapter(lang_adapter, config=lang_adapter_config, with_head=use_la_head)
    model.active_adapters = Stack(la_name, ta_name)

    return model.to(device)

def get_hidden_states(model, input, tokenizer, device):
    model.eval()
    with torch.no_grad():
        input = tokenizer(input, return_tensors='pt')
        output = model(input.input_ids.to(device), attention_mask=input.attention_mask.to(device), output_hidden_states=True)

    # Tuple of  (batch_size, sequence_length, hidden_size)
    return output.hidden_states

def get_all_hidden_states(model, dataset:list, tokenizer, device):
    result = {}

    for data in tqdm(dataset):
        hidden_states = get_hidden_states(model, data, tokenizer, device)

        layer_num = model.config.num_hidden_layers
        for layer_id in range(layer_num + 1):
            vector_in_tensor = hidden_states[layer_id][0].to("cpu")
            vector_in_tensor = torch.mean(vector_in_tensor, dim=0)
            try:
                result[layer_id].append(vector_in_tensor)
            except:
                result[layer_id] = [vector_in_tensor]

    return result

def run_visualization(
            method: str,
            data: List,
            save_dir: str,
            layer: int,
            file_name: str,
            lang_labels: List,
            num_samples_per_lang: int,
            n_components: Optional[int] = None,
            n_neighbors: Optional[int] = None,
            min_dist: Optional[float] = None,
            metric: Optional[str] = None,
            random_state: Optional[int] = None,
            spread: Optional[float] = None,
            ):
    
    inp = torch.stack(data, dim=0).numpy()
    
    if method == "pca":
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(inp)
    
    if method == "umap":
        umap_model = umap.UMAP(n_neighbors=n_neighbors, 
                               min_dist=min_dist, 
                               n_components=n_components,
                               metric=metric,
                               random_state=random_state,
                               spread=spread,
                               )
        result = umap_model.fit_transform(inp)

    plt.cla()

    offsets = np.arange(0, len(inp) + 1, num_samples_per_lang)
    print(offsets)

    #colors = plt.get_cmap('viridis', len(lang_labels))

    colors = {
        'en': '#000000',  # Black
        'es': '#006428',  # Light green 
        'pt': '#278f48',  # Medium green 
        'gl': '#98d594',  # Darker green 
        'ca': '#58b668',  # Dark green
        'de': '#084a91',  # Light blue 94c4df
        'sv': '#2575b7',  # Medium blue 
        'da': '#549fcd',  # Dark blue
        'is': '#94c4df',  # Darker blue
        'nl': '#ffb347',  # Lighter orange
        'af': '#ffeb99',  # Lighter yellow
        'fi': '#9067c6',  # Soft purple 
        'hu': '#b59edb',  # Soft darker purple
    }


    assert len(lang_labels) == (len(offsets) - 1)

    # todo: enable other than 2 components
    for i in range(len(lang_labels)):
        plt.scatter(result[offsets[i]:offsets[i+1], 0], result[offsets[i]:offsets[i+1], 1], label=lang_labels[i], color=colors[lang_labels[i]])

    #plt.legend(fontsize=16)
    plt.legend()

    #save_dir_ = os.path.join(save_dir, f"{method}_figures")
    #os.makedirs(save_dir_, exist_ok=True)
    ##plt.savefig(f"{save_dir_}/{file_name}_{method}_{layer}.png", dpi=300, bbox_inches='tight')

    fn = os.path.join(save_dir, file_name)
    plt.savefig(f"{fn}.png", dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True, help='Base model that has been used for task adapter training')
    parser.add_argument('--hf_token', type=str, required=True, help='Personal HF token to access gated models like Llama 2')
    parser.add_argument('--eval_prefix', type=str, default=None, required=True, help="Unique identifier for saved file")
    parser.add_argument('--task_adapter', type=str, required=False, default=None, help='Path to trained task adapter')
    parser.add_argument('--task_adapter_config', type=str, required=False, default=None, help='Path to task adapter configuration')
    parser.add_argument('--lang_adapter', type=str, required=False, default=None, help='Path to the language adapter for the specified langugae')
    parser.add_argument('--lang_adapter_config', type=str, required=False, default=None, help="Path to language adapter configuration")
    parser.add_argument('--lang_adapter_prefix', type=str, help="The shared prefix of each LA's name")
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory to load model and data from')
    parser.add_argument('--model_cache_dir', type=str, required=False, default=None, help='Separate cache dir for base model')
    parser.add_argument('--overwrite_cache', action="store_true", help="Whether to overwrite the cache")
    parser.add_argument('--source_lang', metavar='N', nargs="+", required=False, default="english", help="Source language of task adapter")
    parser.add_argument('--languages', nargs="+", required=False, default=None, help='Language to evaluate task adapter on')
    parser.add_argument('--language_ratios', type=str, required=False, default="1.0", help="Language ratios of languages to be evaluated")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to evaluate the adapter model on")
    parser.add_argument('--task', type=str, required=False, help='Task to be evaluated')
    parser.add_argument('--task_dataset', type=str, required=False, help='Dataset corresponding to chosen task')
    parser.add_argument('--task_data_split', type=str, required=False, help='Split of chosen dataset to use.')
    parser.add_argument('--eval_size', type=int, required=False, help='Size of subset to evaluate on')
    parser.add_argument('--max_length', type=int, required=False, default=1024, help='Maximum length of input batch')
    parser.add_argument('--max_new_tokens', type=int, required=False, default=100, help='Maximum number of new tokens to be generated')
    parser.add_argument('--save_file', action="store_true", required=False, help="Whether to store the generated predictions and targets")
    parser.add_argument('--use_init', action="store_true", help="Whether to use init-Method to load AdapterModel or do it manually.")
    parser.add_argument('--use_la_head', action="store_true", help="Whether to use the causal LM prediction head of the trained LA")
    parser.add_argument('--use_ta_head', action="store_true", help="Whether to use the causal LM prediction head of the trained TA")
    parser.add_argument('--use_la', action="store_true", help="Whether to activate target LA or not")
    parser.add_argument('--use_ta', action="store_true", help="Whether to activate source TA or not")
    parser.add_argument('--eval_all', action="store_true", help="Whether to evaluate all settings for all languages")
    parser.add_argument('--num_langs', type=int, required=False, default=None, help="Number of languages, only required if 'eval_all' is set")
    parser.add_argument('--setting', type=str, required=False, default=None, help="The evaluation setting. Only required if 'eval_all' flag is set")
    parser.add_argument('--n_shots', type=int, required=False, default=None, help="Modify the number of shots to evaluate")
    parser.add_argument('--fusion_adapter', type=str, required=False, default=None, help="Set if adapter fusion module should be evaluated")
    parser.add_argument('--fuse_langs', metavar='N', type=str, nargs='+', required=False, default=None, help="Which languages to fuse their LAs with each other")
    parser.add_argument('--format_prompt', action="store_true", help="Whether to format instructions in the target language")
    parser.add_argument('--source_prompt', type=str, required=False, default=None, help="Whether to use the instructions in the source language")
    parser.add_argument('--instr_keys', type=str, default=None, help="Whether to use specific instruction keys prepended to the sample")
    parser.add_argument('--save_preproc_data', action="store_true", help="Whether to save the preprocessed dataset (formatted and tokenized)")
    parser.add_argument('--visual_method', type=str, required=True, choices=['pca', 'umap'], help="Method to use for visualizing hidden states.")
    parser.add_argument('--layers', nargs="+", type=int, required=True, default=None, help='Layers to run PCA/UMAP for')
    parser.add_argument('--n_neighbors', type=int, required=False, default=None, help="For UMAP only. The number of neighbors to consider when constructing the local neighborhood for each data point.")
    parser.add_argument('--min_dist', type=float, required=False, default=None, help="For UMAP only. Controls how tightly UMAP packs points in the low-dimensional space.")
    parser.add_argument('--metric', type=str, required=False, default=None, help="For UMAP only. Selects a metric for comparing embeddings.")
    parser.add_argument('--random_state', type=int, required=False, default=None, help="For UMAP only. Random state to reproduce results.")
    parser.add_argument('--spread', type=float, required=False, default=None, help="For UMAP only. Controls the scale of clustering.")
    parser.add_argument('--n_components', type=int, required=True, help="Number of principal components to use for PCA")
    parser.add_argument('--num_samples', type=int, required=False, default=500, help="Number of data samples per language to use for each PCA analysis.")
    parser.add_argument('--path_to_saved_hidden_states', type=str, required=False, default=None, help="Path to a file containing hidden states.")
    parser.add_argument('--save_hidden_states', action="store_true", help="Whether to store the extracted hidden states.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory where to save the PCA plots.")
    parser.add_argument('--num_steps', type=int, required=False, default=None, help="Number of steps used for LA training. Potentially required for building the LA path.")
    parser.add_argument('--adapter_path', type=str, required=False, default=None, help='Path to the directory that stores all trained adapters')
    parser.add_argument('--latest_checkpoint', action="store_true", help="Whether to evaluate the latest checkpoint being saved.")
    parser.add_argument('--seed', type=int, required=False, default=None, help='Seed for TA.')
    parser.add_argument('--ta_path_format', type=str, required=False, default=None, help="Path format for TAs. 'depr' for old Llama 2 TAs, 'old' for non-current ones.")
    parser.add_argument('--merge_weights', action="store_true", help="Whether to merge LoRA weights with the pre-trained weight matrices.")
    parser.add_argument('--adapter_method', type=str, default="bn", choices=["seq_bn", "seq_bn_inv", "prompt_tuning", "lora", "pt"], help="Adapter method used for fine-tuning")
    parser.add_argument('--inlang', action="store_true", help="Use in-language demos for ICL.")
    parser.add_argument('--diff_instr', action="store_true", help="Use different instruction (from MAPLE paper) than standard template")
    parser.add_argument('--no_instr', action="store_true", help="Only add instruction at the top of prompt")
    parser.add_argument('--add_eng_instr', action="store_true", help="Adds English instruction instead of target language")
    parser.add_argument('--add_period', action="store_true", help="Adds period as additional halt signal to samples.")
    parser.add_argument('--lang_specific_indices', type=str, required=False, default=None, help="Path to file containing indices of samples with language-specific targets (for MLQA only)")
    parser.add_argument('--lang_adapter_suffix', type=str, required=False, default=None, help="The shared suffix of each LA's name")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature parameter for generating output. Defaults to 1.0 meaning deterministic output.")
    parser.add_argument('--top_p', type=float, default=1.0, help="Top p sampling parameter for output generation.")   
    parser.add_argument('--seeds', nargs="+", type=int, required=False, default=None, help="List of seeds for TA to be evaluated.")
    parser.add_argument('--stop_strings', metavar='N', nargs="+", type=str, required=False, default=None, help="List of strings to act as stop strings when generating.")
    parser.add_argument('--stop_at_first_upper', action="store_true", help="Whether to cut off generations at first upper-case letter occurred. Only for n-shot and SIB-200")
    parser.add_argument('--xai_method', type=str, required=False, default=None, help="For bash script only.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["HF_TOKEN"] = args.hf_token
    base_model_name_or_path = args.model_name_or_path
    eval_prefix = args.eval_prefix
    task_adapter = args.task_adapter
    task_adapter_config = args.task_adapter_config
    lang_adapter = args.lang_adapter
    lang_adapter_prefix = args.lang_adapter_prefix
    lang_adapter_suffix = args.lang_adapter_suffix
    cache_dir = args.cache_dir
    model_cache_dir = args.model_cache_dir
    overwrite_cache = args.overwrite_cache
    languages = args.languages
    source_lang = args.source_lang
    dataset_name = args.dataset
    task = args.task
    task_dataset = args.task_dataset
    task_data_split = args.task_data_split
    eval_size = args.eval_size
    use_init = args.use_init
    use_la_head = args.use_la_head
    use_ta_head = args.use_ta_head
    use_la = args.use_la
    use_ta = args.use_ta
    n_shots = args.n_shots
    source_prompt = args.source_prompt
    format_prompt = args.format_prompt
    instr_keys = args.instr_keys
    layers = args.layers
    visual_method = args.visual_method
    n_components = args.n_components
    n_neighbors = args.n_neighbors
    min_dist = args.min_dist
    metric = args.metric
    random_state = args.random_state
    spread = args.spread
    num_samples = args.num_samples
    path_to_saved_hidden_states = args.path_to_saved_hidden_states
    save_hidden_states = args.save_hidden_states
    save_dir = args.save_dir

    adapter_path = args.adapter_path
    latest_checkpoint = args.latest_checkpoint
    seed = args.seeds[0]
    merge_weights = args.merge_weights
    ta_path_format = args.ta_path_format
    adapter_method = args.adapter_method 

    inlang = args.inlang
    diff_instr = args.diff_instr
    no_instr = args.no_instr
    add_eng_instr = args.add_eng_instr
    add_period = args.add_period
    lang_specific_indices = args.lang_specific_indices
    temperature = args.temperature
    top_p = args.top_p
    stop_strings = args.stop_strings
    stop_at_first_upper = args.stop_at_first_upper
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")

    if len(source_lang) == 1:
        source_lang = source_lang[0]

    if visual_method == "umap":
        umap_vars = [n_neighbors, min_dist, metric, random_state, spread]
        if any(umap_var is None for umap_var in umap_vars):
            raise ValueError(f"UMAP visualization requires the following parameters: {', '.join(umap_vars)}")

    # for file name formatting
    if "Llama-2" in base_model_name_or_path:
        model_name = "llama-2"
    if "Llama-3" in base_model_name_or_path:
        model_name = "llama-3"
    if "Llama-3.1" in base_model_name_or_path:
        model_name = "llama-31"

    if "sib200" in dataset_name:
        file_data_name = "sib200"
    if "aya" in dataset_name:
        file_data_name = "aya"

    #if not isinstance(setting, str):
    setting = "LA" if use_la else "noLA"

    if isinstance(n_shots, int): eval_prefix = f"icl-{n_shots}-{eval_prefix}"

    run_dir_name = f"{eval_prefix}_{model_name}_{file_data_name}_{num_samples}"
    run_dir = os.path.join(save_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run dir: {run_dir}")

    if save_hidden_states and path_to_saved_hidden_states:
        if not os.path.exists(path_to_saved_hidden_states):
            path_to_saved_hidden_states = None
            print("Extracting hidden states..")
        else:
            print("Loading hidden states..")

    # create or get hidden states
    if path_to_saved_hidden_states is None:
        if (use_ta or use_la) and ("adapters" in active_env) :
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

        
        # optioanlly load ta
        if use_ta and "adapters" in active_env:
            assert isinstance(seed, int), "Requires a seed for task adapter."
            model, task_adapter_path, task_adapter_config, ta_name = load_adapters_ta_model(
                model,
                use_ta,
                use_ta_head,
                use_la,
                adapter_path,
                task_adapter,
                seed,
                latest_checkpoint,
                ta_path_format,
                adapter_method,
                merge_weights,
                )

        if use_ta and "peft" in active_env:
            assert isinstance(seed, int), "Requires a seed for task adapter."
            model, task_adapter_path, ta_name = load_peft_ta_model(
                base_model,
                cache_dir,
                use_la,
                adapter_path,
                task_adapter,
                seed,
                latest_checkpoint,
                ta_path_format,
                merge_weights,
            )

        print(model)
        model.to(device)


        # extract all hidden states for all languages
        hidden_states = {}
        for lang in languages:

            if "adapters" in active_env and use_la:
                '''if multiling:
                    lang_code = f"{'_'.join([LANGUAGE_MAPPING[l] for l in languages])}"
                elif dist_la:
                    if isinstance(one_for_all, str):
                        lang_code = LANGUAGE_MAPPING[one_for_all]
                    else:
                        lang_code = MOST_DIST_LANGS[LANGUAGE_MAPPING[lang]]
                else:
                    lang_code = LANGUAGE_MAPPING[lang]'''
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

                '''if adapter_method == "lora" and merge_weights:
                    model.merge_adapter(la_name)
                    if use_ta:
                        model.merge_adapter(ta_name)'''

                print(model)
                print(model.adapter_summary())
                model.to(device)

            if "peft" in active_env and use_la:
                '''if dist_la:
                    if isinstance(one_for_all, str):
                        la_name =  f"{lang_adapter_prefix}_{LANGUAGE_MAPPING[one_for_all]}_{lang_adapter_suffix}"
                    else:
                        la_name =  f"{lang_adapter_prefix}_{MOST_DIST_LANGS[LANGUAGE_MAPPING[lang]]}_{lang_adapter_suffix}"
                else:
                    la_name =  f"{lang_adapter_prefix}_{LANGUAGE_MAPPING[lang]}_{lang_adapter_suffix}"
                la_path = f"{adapter_path}/{la_name}/{la_name}"'''

                la_name =  f"{lang_adapter_prefix}_{LANGUAGE_MAPPING[lang]}_{lang_adapter_suffix}"
                la_path = f"{adapter_path}/{la_name}/{la_name}"

                print(la_path)

                model = PeftModel.from_pretrained(
                    base_model,
                    la_path,
                    adapter_name=la_name,
                    token=os.environ["HF_TOKEN"],
                    cache_dir=cache_dir,
                    is_trainable=False,
                )

                if use_ta:
                    print("\nNow loading task adapter..\n")
                    model.load_adapter(task_adapter_path, adapter_name=ta_name)
                    model.set_adapter(ta_name)
                
                if adapter_method == "lora" and merge_weights:
                    model.merge_adapter()

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
                                        lang_ratio=1.0,
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
                                            lang_ratio=1.0,
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
                                        )
                    n_shot_examples = n_shot_examples.map(_formatting, load_from_cache_file=not overwrite_cache)
                
                assert isinstance(instr, str)
                n_shot_prompt = instr + "\n\n".join(n_shot_examples['text']) if n_shots > 0 else instr
                assert isinstance(n_shot_prompt, str)
                
                def add_prompt(example):
                    example['text'] = n_shot_prompt + "\n\n" + example['text'] if n_shots > 0 else n_shot_prompt + example['text']
                    return example

                dataset = dataset.map(add_prompt, load_from_cache_file=False) 
            
            if isinstance(lang_specific_indices, str):
                if not os.path.exists(lang_specific_indices):
                    raise ValueError("Specified file path does not exist")

                else:
                    indices = np.load(lang_specific_indices)
                    print(type(indices))
                    print(indices[:10])

                    dataset['test'] = dataset['test'].select(indices)
                    print(dataset['test'][target_column][:10])
                    data_list = dataset['test']['text']
            else:
                data_list = dataset['test']['text']

            print(data_list[0])

            assert len(data_list) >= num_samples, "Selected number of samples exceeds dataset length."
            hidden_states[lang] = get_all_hidden_states(model, data_list[:num_samples], tokenizer, device)
            
            if use_la:
                model.delete_adapter(la_name) 

        if save_hidden_states:
            hidden_dir = os.path.join(save_dir, f"hidden_states_{visual_method}")
            os.makedirs(hidden_dir, exist_ok=True)
            hidden_fn = f"{run_dir_name}_{source_lang}_{setting}_hidden_states.pkl"
            hidden_file_path = os.path.join(hidden_dir, hidden_fn)
            print(hidden_file_path)             
            with open(hidden_file_path, "wb") as f:
                pickle.dump(hidden_states, f)
    
    else:
        with open(path_to_saved_hidden_states, "rb") as f:
            hidden_states = pickle.load(f)

    for layer in layers:
        file_name = f"{run_dir_name}_{layer}_{setting}_{source_lang}"
        print(f"File name: {file_name}")
        
        input_tensor = [
            h_state for lang in languages for h_state in hidden_states[lang][layer][:num_samples]
        ]

        # convert lang codes back to list
        lang_codes = [LANGUAGE_MAPPING[lang] for lang in languages]
        print(f"Languages: {lang_codes}")

        run_visualization(visual_method, 
                        input_tensor, 
                        run_dir, 
                        layer, 
                        file_name, 
                        lang_codes, 
                        num_samples, 
                        n_components, 
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        random_state=random_state,
                        spread=spread,
                        )

if __name__ == "__main__":
    main()