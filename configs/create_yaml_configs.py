from itertools import product
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import ScalarString, SingleQuotedScalarString
from ruamel.yaml.nodes import ScalarNode
import sys
from scripts.utils import get_language_mapping
from configs_arguments import get_ta_train_args, get_ta_eval_args, get_icl_eval_args

config_choices = ["ta_train", "ta_eval", "icl_eval"]
if len(sys.argv) > 1:
    config_setup = sys.argv[1]
else:
    raise ValueError(f"Script requires one of the following config names: {', '.join(config_choices)}")

if len(sys.argv) > 2:
    file_prefix = sys.argv[2]
else:
    file_prefix = None

if len(sys.argv) > 3:
    ta_path = sys.argv[3]
else:
    ta_path = None

if config_setup not in config_choices:
    print(f"Error: '{config_setup}' is not a valid choice.")
    print(f"Valid choices are: {', '.join(config_choices)}")
    sys.exit(1)

# 1. Create a custom MergeKey class
class MergeKey(ScalarString):
    def __new__(cls, value='<<'):
        return ScalarString.__new__(cls, value)

# 2. Define a representer for MergeKey
def merge_key_representer(dumper, data):
    return ScalarNode('tag:yaml.org,2002:merge', '<<')

out_file = sys.argv[1]

yaml = YAML()
yaml.Representer.add_representer(MergeKey, merge_key_representer)

# some string formatting helpers
LANGUAGE_MAPPING = get_language_mapping()

def get_pretty_names(model, ds, adapter_method):
    if "Llama-2" in model:
        model = "llama-2"
    elif "Llama-3" and not "." in model:
        model = "llama-3"
    elif "Llama-3.1" in model:
        model = "llama-31"

    if "sib200" in ds:
        ds = "sib200"
    if "aya" in ds:
        ds = "mlqa"

    if adapter_method == "seq_bn":
        adapter_method = "bn"

    return model, ds, adapter_method.lower()

def get_la_name(model, source_lang, adapter_method):
    prefix = f"{model.split("/")[-1].replace(".", "")}_cc100"

    if adapter_method == "seq_bn":
        suffix = "12500" if not '.' in model else ""
    if adapter_method.lower() == "lora":
        suffix = "lora-alpha"

    if "." in model and adapter_method == "seq_bn":
        return f"{prefix}_{LANGUAGE_MAPPING[source_lang]}", prefix
    else:
        return f"{prefix}_{LANGUAGE_MAPPING[source_lang]}_{suffix}", prefix, suffix

def convert_key(key):
    keys_to_keep = ["hf_token", "lang_ratio"]
    if key in keys_to_keep:
        return False
    
    return True

def scalars_2_strings(data: CommentedMap):
    assert isinstance(data, CommentedMap), "Needs to be of type 'CommentedMap'."

    for key, value in data.items():
        if isinstance(value, str) and convert_key(key):
            data[key] = SingleQuotedScalarString(value)

# creates config maps for ta training
if config_setup == "ta_train":
    defaults, setups, models, datasets, source_languages, lang_ratios, adapter_methods = get_ta_train_args()

    configs = []
    for setup, model, adapter_method, ds, source_language, lang_ratio in product(setups, models, adapter_methods, datasets, source_languages, lang_ratios):
        model_, ds_, adapter_method_ = get_pretty_names(model, ds["dataset_name"], adapter_method["name"])
        if setup == "la":
            adapter_names = {
                                "la_name": get_la_name(model, source_language, adapter_method["name"])[0],
                                "ta_name": f"{model_}_{adapter_method_}_cc100_{LANGUAGE_MAPPING[source_language]}_{ds_}"
                            }
            
            adapter_names.update({"merge_weights": ""} if adapter_method_ == "lora" else {})

        else:
            adapter_names = {
                                "ta_name": f"{model_}_{adapter_method_}_{LANGUAGE_MAPPING[source_language]}_{ds_}"
                            }       
                            
        config = {
            MergeKey(): defaults,
            "model_name_or_path": model,
            "per_device_train_batch_size": 2 if "3" in model and adapter_method_ == "lora" else 4,
            "per_device_eval_batch_size": 2 if "3" in model and adapter_method_ == "lora" else 4,
            "ta_languages": source_language,
            "lang_ratio": lang_ratio,
            **{k: v for k, v in ds.items()},
            **{k: v for k, v in adapter_method.items() if k != "name"},
            **{k: v for k, v in adapter_names.items()}
        }

        config_map = CommentedMap(config)
        scalars_2_strings(config_map)
        configs.append(config_map)

if config_setup == "ta_eval":
    defaults, setups, models, datasets, source_languages, adapter_methods = get_ta_eval_args()

    configs = []
    for setup, model, adapter_method, ds, source_language in product(setups, models, adapter_methods, datasets, source_languages):
        model_, ds_, adapter_method_ = get_pretty_names(model, ds["dataset"], adapter_method)
        if setup == "la":
            adapter_names = {
                                "use_ta": "",
                                "task_adapter": f"{model_}_{adapter_method_}_cc100_{LANGUAGE_MAPPING[source_language]}_{ds_}",
                                "use_la": "",
                                "lang_adapter_prefix": get_la_name(model, source_language, adapter_method)[1],
                            }
            #adapter_names.update({"ta_path_format": ta_path if ta_path else "final"})
            adapter_names.update({} if "." in model and adapter_method == "seq_bn" else {"lang_adapter_suffix": get_la_name(model, source_language, adapter_method)[2]})

        else:
            adapter_names = {
                                "use_ta": "",
                                "task_adapter": f"{model_}_{adapter_method_}_{LANGUAGE_MAPPING[source_language]}_{ds_}",
                            }
            #adapter_names.update({"ta_path_format": "ta_final_sib200-drop" if ds_ == "sib200" else "ta_final"})
        
        # add specific file path
        #adapter_names.update({"ta_path_format": ta_path if ta_path else "final"})
        adapter_names.update({"merge_weights": ""} if adapter_method == "lora" else {})
            
        config = {
            MergeKey(): defaults,
            "model_name_or_path": model,
            "source_lang": source_language,
            "adapter_method": adapter_method,
            "eval_prefix": f"{adapter_method_}-{ta_path.split("-")[-1]}" if isinstance(ta_path, str) and len(ta_path.split("-")) > 1 else adapter_method_,
            **{k: v for k, v in ds.items()},
            **{k: v for k, v in adapter_names.items()}
        }

        config_map = CommentedMap(config)
        scalars_2_strings(config_map)
        configs.append(config_map)


# icl eval args
if config_setup == "icl_eval":
    defaults, setups, models, datasets, source_languages, adapter_methods = get_icl_eval_args()

    configs = []
    for setup, model, adapter_method, ds, source_language in product(setups, models, adapter_methods, datasets, source_languages):
        model_, ds_, adapter_method_ = get_pretty_names(model, ds["dataset"], adapter_method)
        if setup == "la":
            adapter_names = {
                                "use_la": "",
                                "lang_adapter_prefix": get_la_name(model, source_language, adapter_method)[1]
                            }
            adapter_names.update({} if "." in model and adapter_method == "seq_bn" else {"lang_adapter_suffix": get_la_name(model, source_language, adapter_method)[2]})
            adapter_names.update({"merge_weights": ""} if adapter_method == "lora" else {})
                            
        config = {
            MergeKey(): defaults,
            "model_name_or_path": model,
            "source_lang": source_language,
            "adapter_method": adapter_method,
            "format_prompt": "",
            "source_prompt": "english",
            "no_instr": "",
            "add_eng_instr": "",
            "add_period": "",
            "eval_prefix": f"icl_{defaults['n_shots']}_{adapter_method_}",
            **{k: v for k, v in ds.items()},
            #**{k: v for k, v in adapter_names.items()}
        }
        config.update(adapter_names if setup == "la" else {})

        config_map = CommentedMap(config)
        scalars_2_strings(config_map)
        configs.append(config_map)

defaults.yaml_set_anchor('default_values')
scalars_2_strings(defaults)

configs = CommentedSeq(configs)
for i in range(len(configs)):
    comment_text = f"\n{i}"
    configs.yaml_set_comment_before_after_key(i, before=comment_text)

# Combine defaults and configs into a single YAML structure
final_yaml = CommentedMap({
    "defaults": defaults,
    "configs": configs
})

final_yaml.yaml_set_comment_before_after_key("configs", before="\n")

# Write to YAML file
fn = f"all_{file_prefix}_{config_setup}_configs.yaml.j2" if isinstance(file_prefix, str) else f"all_{config_setup}_configs.yaml.j2"
with open(fn, 'w') as f:
    yaml.dump(final_yaml, f)

