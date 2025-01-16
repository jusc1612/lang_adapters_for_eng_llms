import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional
from datasets import load_dataset
from math import gcd
from functools import reduce
import datasets
from huggingface_hub import HfApi
import os

def main_process_print(message, rank: int):
    """
    Prints a message only if the current process is the main process.
    
    Args:
        message (str): The message to print.
        rank (int): The rank of the current process.
    """
    if rank == 0:
        print(message)

def floats_to_ints(floats):
    # Scale up the floats to preserve decimal precision
    N = 1000000
    integer_counts = [round(f * N) for f in floats]
    
    # Compute the greatest common divisor (GCD) of the integer counts
    the_gcd = reduce(gcd, integer_counts)
    
    # Simplify the integer counts by dividing by the GCD
    integer_counts = [i // the_gcd for i in integer_counts]
    
    return integer_counts

def string_to_list(string: str):
  """Converts a string of numbers separated by spaces into a list of floats.

  Args:
    string: The input string.

  Returns:
    A list of floats, or None if the conversion fails.
  """

  try:
    float_list = [float(num) for num in string.split()]
    return float_list
  
  except ValueError:
    return string.split()

def count_lines(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for line in file)
        print(f"Number of samples: {line_count}")

def get_hf_dataset_files(
        ds: str,
        lang: str,
        max_number: int,
        train_file_pattern: Optional[str] = None,
        file_extension: Optional[str] = None,
    ):

    if "CulturaX" in ds:
        train_file_pattern = f"{lang}_part"
        file_extension = ".parquet"

    api = HfApi()
    files = api.list_repo_files(repo_id=ds, repo_type='dataset')

    lang_files = [f for f in files if train_file_pattern in f and f.endswith(file_extension)]

    lang_files = lang_files[:max_number] if len(lang_files) > max_number else lang_files
    print(f"Language: {lang}\nData files: {lang_files}")

    return lang_files

def get_raw_dataset(
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        preprocessing_num_workers: Optional[int] = None,
        validation_split_percentage: Optional[int] = None,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        keep_linebreaks: Optional[bool] = False,
    ):

    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            num_proc=preprocessing_num_workers,
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                data_files=data_files,
                split=f"train[:{validation_split_percentage}%]",
                cache_dir=cache_dir,
                num_proc=preprocessing_num_workers,
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                data_files=data_files,
                split=f"train[{validation_split_percentage}%:]",
                cache_dir=cache_dir,
                num_proc=preprocessing_num_workers,
            )

    else:
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
            #use_auth_token=True if use_auth_token else None,
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
                #use_auth_token=True if use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{validation_split_percentage}%:]",
                cache_dir=cache_dir,
                num_proc=preprocessing_num_workers,
                #use_auth_token=True if use_auth_token else None,
                **dataset_args,
            )
    
    return raw_datasets

def get_language_mapping():
    mapping = {"afrikaans": "af", 
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
               "swedish": "sv"
                }

    return mapping

def code_2_lang():
    mapping = {
            "af": "afrikaans",
            "ca": "catalan",
            "da": "danish",
            "de": "german",
            "en": "english",
            "fi": "finnish",
            "gl": "galician",
            "hu": "hungarian",
            "is": "icelandic",
            "nl": "dutch",
            "pt": "portuguese",
            "es": "spanish",
            "sv": "swedish"
        }

    return mapping

def get_translations():
    translations = {
        'english': ('Passage', 'Question', 'Answer'),
        'german': ('Passage', 'Frage', 'Antwort'),
        'swedish': ('Passage', 'Fråga', 'Svar'),
        'spanish': ('Pasaje', 'Pregunta', 'Respuesta'),
        'dutch': ('Passage', 'Vraag', 'Antwoord'),
        'portuguese': ('Passagem', 'Pergunta', 'Resposta'),
        'catalan': ('Passatge', 'Pregunta', 'Resposta'),
        'finnish': ('Kappale', 'Kysymys', 'Vastaus'),
        'hungarian': ('Részlet', 'Kérdés', 'Válasz'),
        'danish': ('Passage', 'Spørgsmål', 'Svar'),
        'icelandic': ('Kafli', 'Spurning', 'Svar'),
        'galician': ('Pasaxe', 'Pregunta', 'Resposta'),
        'afrikaans': ('Deel', 'Vraag', 'Antwoord')
    }
    
    return translations

def get_sib200_en_topics():
    return ['science/technology', 'travel', 'politics', 'sports', 'health', 'entertainment', 'geography']

def get_mlqa_instr():
    instr = (
    "### Instruction: "
    "The task is to solve reading comprehension problems. "
    "You will be provided questions on a set of passages and you will need to provide the answer as it appears in the passage. "
    "The answer should be in the same language as the question and the passage. "
    "Provide nothing else beyond the answer.\n\n"
    )
    return instr

def get_source_lang_instr_mlqa(source_lang):
    source_lang_instrs = {
        "english": "Refer to the passage below and then answer the question afterwards in the same language as the passage:",
        "german": "Verweisen Sie auf die nachstehende Passage und beantworten Sie anschließend die Frage in derselben Sprache wie die Passage:",
        "spanish": "Consulte el pasaje siguiente y luego responda a la pregunta posteriormente en el mismo idioma que el pasaje:"
    }

    if source_lang not in source_lang_instrs.keys():
        raise ValueError(f"Currently only {', '.join(list(source_lang_instrs.keys()))} are supported source languages.")
    
    return source_lang_instrs[source_lang]

def get_sib200_instr():
    instr = "Classify the following sentence into one of the following topics:\n1. science/technology\n2. travel\n3. politics\n4. sports\n5. health\n6. entertainment\n7. geography\n\n"
    return instr

def get_most_dist_lang():
    most_dist_langs = {
        "en": "hu",
        "de": "hu",
        "sv": "hu",
        "da": "gl",
        "af": "hu",
        "nl": "fi",
        "is": "hu",
        "es": "af",
        "gl": "hu",
        "pt": "af",
        "ca": "hu",
        "fi": "gl",
        "hu": "gl",
    }
    
    return most_dist_langs

def get_latest_checkpoint(adapter_dir):

    checkpoints = [d for d in os.listdir(adapter_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(adapter_dir, d))]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {adapter_dir}")

    checkpoint_numbers = [int(d.split("-")[-1]) for d in checkpoints]
    
    latest_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"

    print(f"Latest checkpoint: {latest_checkpoint}")
    
    return os.path.join(adapter_dir, latest_checkpoint)

def formatting(sample, tokenizer, dataset, train=False, n_shot=False, instr_keys=None, lang=None, source=None, no_instr=False, add_period=False, keep_en_instr=False):

    if 'sib200' in dataset:
        sib_trans = pd.read_csv('/home/jschlenker/thesis_experiments/translations/sib_200_prompts_translated.csv')
        input_column = 'text'
        target_column = 'category'
        
        if instr_keys == "chat":
            instruction = "### Human: "
            INPUT_KEY = ""
            RESPONSE_KEY = "### Assistant:"
        else:
            instruction = ""
            INPUT_KEY = ""
            RESPONSE_KEY = "Topic:"
        
        if source:
            assert isinstance(source, str)
            lang_prompt = sib_trans.loc[sib_trans['language'] == source.capitalize()].values[0]

            if no_instr and n_shot:
                instruction = None
                INPUT_KEY =  f"{lang_prompt[2].capitalize()}:"
                RESPONSE_KEY = RESPONSE_KEY if instr_keys == "chat" else f"{lang_prompt[4].capitalize()}:"
            
            # todo: modify such that it enables more source languages than English
            else:
                #instruction = f"{instruction}Classify the following sentence into one of the following topics:\n1. science/technology\n2. travel\n3. politics\n4. sports\n5. health\n6. entertainment\n7. geography"
                # only labels in so
                if keep_en_instr:
                    instruction = f"{instruction}Classify the following sentence into one of the following topics:\n{lang_prompt[3]}"
                    INPUT_KEY = "Sentence:"
                    RESPONSE_KEY = RESPONSE_KEY if instr_keys == "chat" else "Topic:"

                else:
                    # at inference: instr+label in non-English source
                    instruction = f"{instruction}{lang_prompt[1]}\n{lang_prompt[3]}"
                    INPUT_KEY =  f"{lang_prompt[2].capitalize()}:"
                    RESPONSE_KEY = RESPONSE_KEY if instr_keys == "chat" else f"{lang_prompt[4].capitalize()}:"
                
                topics = re.sub(r'\d+\.\s*', '', lang_prompt[3]).split()
                topics_en = get_sib200_en_topics()
                topics_dict = {topic_en: topic for topic_en, topic in zip(topics_en, topics)}
                sample[target_column] = topics_dict[sample[target_column]]
                                  
        else:
            assert isinstance(lang, str), "This setup requires a target language to select the respecive prompt."

            lang_prompt = sib_trans.loc[sib_trans['language'] == lang.capitalize()].values[0]
            if keep_en_instr:
                instruction = f"{instruction}Classify the following sentence into one of the following topics:\n{lang_prompt[3]}"
                INPUT_KEY = "Sentence:"
                RESPONSE_KEY = RESPONSE_KEY if instr_keys == "chat" else "Topic:"
            else:
                # get instruction + labels in target lang
                instruction = f"{instruction}{lang_prompt[1]}\n{lang_prompt[3]}"

                # get sentence + topic in target lang
                INPUT_KEY = f"{lang_prompt[2].capitalize()}:"
                RESPONSE_KEY = RESPONSE_KEY if instr_keys == "chat" else f"{lang_prompt[4].capitalize()}:"

            # tranlsate target from English into target language 
            topics = re.sub(r'\d+\.\s*', '', lang_prompt[3]).split()
            topics_en = get_sib200_en_topics()
            topics_dict = {topic_en: topic for topic_en, topic in zip(topics_en, topics)}
            sample[target_column] = topics_dict[sample[target_column]]
        
        '''else:
            instruction = "Classify the following sentence into one of the following topics:\n1. science/technology\n2. travel\n3. politics\n4. sports\n5. health\n6. entertainment\n7. geography"
            INPUT_KEY = "Sentence:"
            RESPONSE_KEY = "Topic:"
            #instruction = "### Human: Classify the following sentence into one of the following topics:\n1. science/technology\n2. travel\n3. politics\n4. sports\n5. health\n6. entertainment\n7. geography"
            #INPUT_KEY = "Sentence:"
            #RESPONSE_KEY = "### Assistant "     '''   

    if 'aya' in dataset:
        translations = get_translations()
        instruction = None
        input_column = 'inputs'
        target_column = 'targets'

        if instr_keys == "chat":
            INPUT_KEY = "### Human:"
            RESPONSE_KEY = "### Assistant:"
        
        elif instr_keys == "icl":
            INPUT_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"

        elif instr_keys == "engl_answer":
            if not source:
                raise ValueError("Invalid combination: Format your instruction according to the source language using the 'source' argument")
            INPUT_KEY = "Instruction:"
            RESPONSE_KEY = "Answer:"

        elif instr_keys == "answer":
            assert isinstance(lang, str)
            if source:
                raise ValueError("Invalid combination: Format your instruction according to the target language by omitting the 'source' argument")
            INPUT_KEY = ""
            RESPONSE_KEY = f"{translations[lang][2]}: "

        elif isinstance(instr_keys, str):
            raise ValueError("Invalid instruction template. Please, choose one of 'chat', 'engl_answer' and 'answer'.")
        
        else:
            INPUT_KEY = ""
            RESPONSE_KEY = ""
    #END_KEY = tokenizer.eos_token

    # Combine a prompt with the static strings
    input_context = f"{INPUT_KEY} {sample[input_column]}"
    if train:
        response = f"{RESPONSE_KEY} {sample[target_column]}"
        if n_shot:
            response = response + "." if add_period else response
        else:
            response += tokenizer.eos_token
    
    else:
        response = f"{RESPONSE_KEY}"
    #end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [instruction, input_context, response] if part]

    # Join prompt template elements into a single string to create the prompt template
    if no_instr and "sib200" in dataset:
        formatted_prompt = "\n".join(parts)
    else:
        formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample

def formatting_(sample, tokenizer, dataset, train=False):
    if 'sib200' in dataset:
        instruction = "Classify the following sentence into one of the following topics:\n1. science/technology\n2. travel\n3. politics\n4. sports\n5. health\n6. entertainment\n7. geography"
        INPUT_KEY = "Sentence:"
        RESPONSE_KEY = "Topic:"
        input_column = 'text'
        target_column = 'category'

    if 'aya' in dataset:
        instruction = None
        INPUT_KEY = "### Human:"
        RESPONSE_KEY = "### Assistant:"
        input_column = 'inputs'
        target_column = 'targets'
    

    input_context = f"{INPUT_KEY} {sample[input_column]}"
    if train:
        response = f"{RESPONSE_KEY} {sample[target_column]}" + tokenizer.eos_token
    else:
        response = f"{RESPONSE_KEY}"

    parts = [part for part in [instruction, input_context, response] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)
    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt
    return sample

def preprocess_batch(batch, tokenizer, max_length, column="text"):
    '''if return_tensors:
        return tokenizer(
                batch["text"],
                max_length = max_length,
                truncation = True,
                return_tensors='pt',
            )'''     
    return tokenizer(
        batch[column],
        max_length=max_length,
        truncation=True,
    )

# Evaluation
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

# taken from https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py#L66 for SQuAD Evaluation
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(preds, labels):
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    #labels = labels[:, 1:].reshape(-1)
    #preds = preds[:, :-1].reshape(-1)

    assert len(labels) == len(preds)

    # exact match
    #em.compute(predictions=preds, references=labels)
    em = sum(compute_exact(label, pred) for label, pred in zip(labels, preds)) / len(labels)

    # f1 score
    f1_scores = np.array([compute_f1(label, pred) for label, pred in zip(labels, preds)])
    avg_f1 = np.mean(f1_scores)

    return {"em": round(em, 2), "f1": round(avg_f1, 2)}

# deprecated function to create multilingual dataset
def concatenate_multiling_iter(ds_dict: dict, language_ratios: dict, batch_size: int):
    langs = list(ds_dict.keys())
    language_iters = {lang: iter(ds_dict[lang]) for lang in langs}

    language_steps = {lang: int(ratio * 10) * batch_size for lang, ratio in language_ratios.items()}
    combined_list = []

    while language_iters:
        for lang, iterator in language_iters.items():
            num_samples = language_steps[lang]
            try:
                batch = [next(iterator) for _ in range(num_samples)]
                combined_list.extend(batch)
            except StopIteration:
                # removes respective iterator as soon as it is exhausted; alternative: start anew 
                #language_iters.remove(iterator)
                #del language_iters[lang]
                print(f"{lang} removed")
                lang_to_delete = lang
                language_iters = {lang: iterator for lang, iterator in language_iters.items() if lang != lang_to_delete}

    return datasets.Dataset.from_list(combined_list)
