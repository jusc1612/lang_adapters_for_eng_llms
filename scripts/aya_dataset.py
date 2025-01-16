from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
import random
from typing import Optional, Union, List
from functools import partial
from transformers import set_seed
import numpy as np
import os

from utils import get_source_lang_instr_mlqa, get_language_mapping, get_translations


ENGL_INSTR = "Refer to the passage below and then answer the question afterwards in the same language as the passage:"
ICL_ENGL_INSTR = (
    "The task is to solve reading comprehension problems. "
    "You will be provided questions on a set of passages and you will need to provide the answer as it appears in the passage. "
    "The answer should be in the same language as the question and the passage."
)
ICL_Q = "Referring to the passage above, the correct answer to the given question is"

Q_WORDS = ["question", "frage", "pregunta", "vraag", "pergunta", "kérdés", "spørgsmål", "spurning", "kysymys", "fråga"]

random_seed = 42
set_seed(random_seed)

TRANSLATIONS = get_translations()
LANGUAGE_MAPPING = get_language_mapping()

def check_uniform_length(datasets: list) -> int:
    assert len(datasets) > 0
    first_length = len(datasets[0])
    
    assert all(len(dataset) == first_length for dataset in datasets), "Not all lists have the same length"

    return first_length

def filter_by_task_and_data(example, task, task_data, task_data_split):
    return example['task_type'] == task and example['sub_dataset_name'] == task_data_split and example['dataset_name'] in task_data

def add_format(sample, source=None, lang=None, n_shot=False, diff_instr=False, no_instr=False):
    example = sample['inputs']
    
    assert ":" in example
    #example = re.split(r'[.:]', example)
    example = example.split(':')
    example = [sent.strip() for sent in example]
    
    INSTR = f"{example[0]}:"

    if len(example[1].split()) == 1:
        passage_start = 2
    else:
        passage_start = 1 

    if len(example) > passage_start + 1:
        passage = example[passage_start:-1]
        passage =': '.join(passage)
    else: 
        tmp = example[passage_start].split('.')
        q = tmp[-1].strip()
        passage = ". ".join(tmp[:-1]) + "."
        example = example[:passage_start] + [passage] + [q]

    qword = passage.split()[-1].strip()
    if qword.lower() in Q_WORDS:
        passage = " ".join(passage.split()[:-1])

    sep = "\n" if n_shot else "\n\n"

    if isinstance(source, str):
        passage = f"{sep}{TRANSLATIONS[source][0]}: {passage}"
        q = f"{sep}{TRANSLATIONS[source][1]}: {example[-1]}"
        if no_instr:
            sample['inputs'] = passage + q
        elif diff_instr:
            sample['inputs'] = ICL_ENGL_INSTR + passage + q + "\n\n" + ICL_Q
        else:
            sample['inputs'] = get_source_lang_instr_mlqa(source) + passage + q
    
    else:
        assert isinstance(lang, str)

        passage = f"{sep}{TRANSLATIONS[lang][0]}: {passage}"
        q = f"{sep}{TRANSLATIONS[lang][1]}: {example[-1]}"

        if no_instr:
            sample['inputs'] = passage + q
        else:
            sample['inputs'] = INSTR + passage + q

    return sample

def make_dataset(dataset: str, 
                 mode: str, 
                 languages: Union[str, List[str]],
                 lang_ratio: Union[float, List[float]],  
                 cache_dir: str,
                 overwrite_cache: Optional[bool] = False, 
                 train_size: Optional[int] = None, 
                 eval_size: Optional[int] = None, 
                 task: Optional[str] = 'question-answering', 
                 task_data: Optional[str] = 'MLQA-en (T)',
                 task_data_split: Optional[str] = 'sentence_split',
                 english_format: Optional[bool] = False,
                 source_prompt: Optional[str] = None,
                 save_file: Optional[bool] = False,
                 data_seed: Optional[int] = 42,
                 n_shot: Optional[bool] = False,
                 diff_instr: Optional[bool] = False,
                 no_instr: Optional[bool] = False,
                 ):

    '''parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True, choices=['comps', 'sttn'], help='Which dataset to use')
    parser.add_argument('--prem_hyp', action="store_true", required=False, help='For COMPS only: Split data into labeled Premise and Hypothesis')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], help='Set training or evaluation mode')
    parser.add_argument('--language', type=str, required=True, help='Language to train/evaluate task adapter for')
    parser.add_argument('--size', type=int, required=False, help='Dataset size. Uses full dataset if not specified.')
    
    args = parser.parse_args()
    print(args)'''

    # CohereForAI/aya_collection_language_split
    
    languages = [languages] if isinstance(languages, str) else languages
    lang_ratio = [lang_ratio] if isinstance(lang_ratio, float) else lang_ratio
    
    if mode in ['train_eval', 'train'] and not all(lang in TRANSLATIONS.keys() for lang in languages):
        raise ValueError(
                "Other languages than set of target languages are not supported by this script for training a task adapter."
            )
    if mode == "eval" and len(languages) > 1:
        raise ValueError(
            "Currently, this script only supports evaluating one language at a time."
        )

    assert len(languages) == len(lang_ratio)
    assert sum(lang_ratio) == 1, "Language ratios must sum to 1."

    task_data=[task_data] if isinstance(task_data, str) else task_data

    datasets = {}
    for lang in languages: 
        data_files = {"test":f"{lang}/test-00000-of-00001.parquet"}
        dataset_lang = load_dataset(f"{dataset}", 
                                    data_files=data_files, 
                                    cache_dir=cache_dir,
                                    token=os.getenv('HF_TOKEN', None),
                                    #download_mode='force_redownload',
                                    )
        
        #dataset = dataset.map(lambda example, idx: {'index': idx}, with_indices=True)

        partial_filter = partial(filter_by_task_and_data, task=task, task_data=task_data, task_data_split=task_data_split)
        dataset_lang = dataset_lang.filter(partial_filter, load_from_cache_file=not overwrite_cache)

        if english_format:
            partial_format = partial(add_format, source=source_prompt, lang=lang, n_shot=n_shot, diff_instr=diff_instr, no_instr=no_instr)
            dataset_lang = dataset_lang.map(partial_format, load_from_cache_file=not overwrite_cache)

        datasets[lang] = dataset_lang["test"]
    
    print(datasets)

    _ = check_uniform_length(list(datasets.values()))

    # determine indices based on the first language in the datasets dict since all datasets have the same length 
    train_indices, test_indices = train_test_split(
        np.arange(len(next(iter(datasets.values())))), test_size=0.2, random_state=data_seed
    ) 
    
    if len(languages) > 1 and "train" in mode:
        #total_data_size = check_uniform_length(list(datasets.values()))
        total_train_size = len(train_indices)
        samples_per_language = {lang: int(total_train_size * ratio) for lang, ratio in zip(languages, lang_ratio)}

        print(samples_per_language)
        
        # adjust the first language to match the total size in case the sampled size does not match the total_data_size
        total_samples = sum(samples_per_language.values())
        if total_samples < total_train_size:
            first_language = languages[0]
            samples_per_language[first_language] += total_train_size - total_samples
        
        assert sum(samples_per_language.values()) == total_train_size

        # get indices and shuffle them to ensure that each sample is selected only once across languages 
        sample_indices = np.arange(total_train_size)
        np.random.seed(random_seed)
        np.random.shuffle(sample_indices)

        split_indices = np.cumsum(list(samples_per_language.values()))[:-1]
        indices_per_language = np.split(sample_indices, split_indices)

        train_datasets = {}
        for i, (lang, dataset) in enumerate(datasets.items()):
            train_datasets[lang] = dataset.select(indices_per_language[i])

        # note: not shuffled, consider shuffling data or returning non-concatenated dataset to further arrange language order after packing
        # in previous implementation: data is shuffled since train, test indices are determined afterwards 
        train_dataset = concatenate_datasets(list(train_datasets.values()))
    else:
        train_dataset = datasets[languages[0]].select(train_indices)

    #dataset = DatasetDict({"validation": concatenate_datasets(list(datasets.values()))})
    #print(dataset)

    if save_file:
        dataset['test'].to_pandas().to_csv("/netscratch/jschlenker/eval/test_aya.csv", index=False)

    '''train_indices, test_indices = train_test_split(
        np.arange(len(dataset['validation'])), test_size=0.2, random_state=random_seed
    )'''

    #train_dataset = dataset['validation'].select(train_indices)
    #eval_dataset = dataset['validation'].select(test_indices)

    # todo: better solution for multilingual ta
    eval_dataset = datasets[languages[0]].select(test_indices)

    if train_size is None and eval_size is None and 'train' in mode:   
        return DatasetDict({"train": train_dataset, "test": eval_dataset})

    elif train_size is None and eval_size is None and mode == 'train':
        return DatasetDict({"train": train_dataset})
    
    elif train_size is None and eval_size is None and mode == 'eval':
        return DatasetDict({"test": eval_dataset})
    
    else:
        random.seed(random_seed)
        
        # for few-shot setting
        if train_size is not None:
            random_indices = random.sample(range(len(train_dataset)), train_size)
            train_dataset = train_dataset.select(random_indices)
            if mode == "train":
                return DatasetDict({"train": train_dataset})             
        
        random.seed(random_seed)
        if eval_size is not None:
            random_indices = random.sample(range(len(eval_dataset)), eval_size)
            eval_dataset = eval_dataset.select(random_indices)
            if mode == "eval":
                return DatasetDict({"test": eval_dataset})

        dataset = DatasetDict({"train": train_dataset, "test": eval_dataset})

        return dataset

#if __name__ == "__main__":
#    main()

