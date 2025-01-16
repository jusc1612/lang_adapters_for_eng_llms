from ruamel.yaml.comments import CommentedMap

def get_ta_train_args():
    """
    For task adapter training as implemented in this thesis, this function returns all arguments required to train all task adapter configurations.
    """

    defaults = CommentedMap({
        "hf_token": "{{ env('HF_TOKEN', None) }}",
        "cache_dir": "{{ env('CACHE_DIR', None) }}",
        "output_dir": "{{ env('OUTPUT_DIR', None) }}",
        "ta_path": "ta_final_sib200-drop",
        "mode": "train",
        "do_train": "",
        "overwrite_output_dir": "",
        "ddp_backend": "nccl",
        "task": "question-answering",
        "task_dataset": "MLQA-en",
        "task_dataset_split": "sentence_split",
        "log_level": "info",
        "report_to": "wandb",
        "data_seed": 42,
        "do_eval": "",
        "eval_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": "",
    })

    setups = ["la", "nola"]

    models = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B"]

    datasets = [
                {
                    "dataset_name": "CohereForAI/aya_collection_language_split",
                    "num_train_epochs": 3, 
                    "instr_keys": "chat",
                    "english_format": "",
                    "source_prompt": "english",
                    "save_steps": 100,
                    "eval_steps": 100,
                    "logging_steps": 100,
                },
                {
                    "dataset_name": "Davlan/sib200",
                    "num_train_epochs": 20, 
                    "english_format": "",
                    "source_prompt": "english",
                    "save_steps": 20,
                    "eval_steps": 20,
                    "logging_steps": 20,            
                }
    ]

    source_languages = ["english", "german", "spanish"]
    lang_ratios = ["1.0"]

    adapter_methods = [
                        {
                            "name": "seq_bn",
                            "adapter_config": "seq_bn",
                            "train_adapter": "",
                            "reduction_factor": 32,
                            "adapter_dropout": 0.1,
                        },
                        {
                            "name": "lora",
                            "peft_type": "LORA",
                            "task_type": "CAUSAL_LM",
                            "lora_rank": 64,
                            "lora_alpha": 32,
                            "adapter_dropout": 0.1,
                            "attn_matrices": "q_proj v_proj k_proj",
                        }
                        ]

    return defaults, setups, models, datasets, source_languages, lang_ratios, adapter_methods

def get_ta_eval_args():

    defaults = CommentedMap({
        "hf_token": "{{ env('HF_TOKEN', None) }}",
        "cache_dir": "{{ env('CACHE_DIR', None) }}",
        "adapter_path": "{{ env('ADAPTER_PATH', None) }}",
        "eval_dir": "{{ env('EVAL_DIR', None) }}",
        "language_ratios": "1.0",
        "max_length": 4096,
        "use_init": "",
        "task": "question-answering",
        "task_dataset": "MLQA-en",
        "task_data_split": "sentence_split",
        "temperature": 0.6,
        "top_p": 0.9,
        "seeds": "42 43 44 45 46",
        "languages": "english german dutch swedish danish icelandic afrikaans spanish portuguese galician catalan finnish hungarian",
        "eval_all": "",
    })

    setups = ["la", "nola"]

    models = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B"]

    datasets = [
                {
                    "dataset": "CohereForAI/aya_collection_language_split",
                    "max_new_tokens": 120,
                    "instr_keys": "chat",
                    "format_prompt": "",
                    "source_prompt": "english",
                    "ta_path_format": "ta_final",
                },
                {
                    "dataset": "Davlan/sib200",
                    "max_new_tokens": 20,
                    "format_prompt": "",
                    "source_prompt": "english",
                    "stop_strings": '### Sentence Topic .',
                    "stop_at_first_upper": "",
                    "latest_checkpoint": "",
                    "ta_path_format": "ta_final_sib200-drop",            
                }
                ]

    source_languages = ["english", "german", "spanish"]

    adapter_methods = ["seq_bn", "lora"]
    
    return defaults, setups, models, datasets, source_languages, adapter_methods

def get_icl_eval_args():
    defaults = CommentedMap({
        "hf_token": "{{ env('HF_TOKEN', None) }}",
        "cache_dir": "{{ env('CACHE_DIR', None) }}",
        "adapter_path": "{{ env('ADAPTER_PATH', None) }}",
        "eval_dir": "{{ env('EVAL_DIR', None) }}",
        "language_ratios": "1.0",
        "max_length": 4096,
        "use_init": "",
        "task": "question-answering",
        "task_dataset": "MLQA-en",
        "task_data_split": "sentence_split",
        "temperature": 0.6,
        "top_p": 0.9,
        "seeds": "42",
        "languages": "english german dutch swedish danish icelandic afrikaans spanish portuguese galician catalan finnish hungarian",
        #"inlang":"",
        "eval_all": "",
        "n_shots": 5,
        "stop_strings": "### Sentence Topic .",
        "stop_at_first_upper": "",
    })

    setups = ["nola", "la"]

    models = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B"]

    datasets = [
                {
                    "dataset": "CohereForAI/aya_collection_language_split",
                    "max_new_tokens": 120,
                    "instr_keys": "chat",
                },
                {
                    "dataset": "Davlan/sib200",
                    "max_new_tokens": 20,         
                }
                ]
    
    # for zero shot ICL
    #source_languages = ["english"]
    
    source_languages = ["english", "german", "spanish"]

    adapter_methods = ["seq_bn", "lora"]

    return defaults, setups, models, datasets, source_languages, adapter_methods

