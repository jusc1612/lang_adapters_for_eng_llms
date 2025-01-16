from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import set_seed
from typing import Optional, List, Union
import random
from utils import string_to_list

seed = 42
set_seed(seed)

LANGUAGE_MAPPING = {"afrikaans": "afr_Latn", 
                    "catalan": "cat_Latn", 
                    "danish": "dan_Latn", 
                    "german": "deu_Latn", 
                    "english": "eng_Latn", 
                    "faroese": "fao_Latn", 
                    "finnish": "fin_Latn", 
                    "galician": "glg_Latn", 
                    "hungarian": "hun_Latn", 
                    "icelandic": "isl_Latn", 
                    "dutch": "nld_Latn", 
                    "portuguese": "por_Latn", 
                    "spanish": "spa_Latn", 
                    "swedish": "swe_Latn"}

def make_dataset(dataset: str, 
                 mode: str, 
                 language: Union[List[str], str],  
                 cache_dir: str, 
                 train_size: Optional[int] = None, 
                 eval_size: Optional[int] = None, 
                 data_seed: Optional[int] = 42,
                 ):
    
    '''if mode == 'train' and language != 'english':
        raise ValueError(
                "Other languages than English are not supported by this script for training a task adapter."
            )'''
    

    if isinstance(language, List):
        assert len(language) == 1, "Currently, this script doesn't support multilingual training. Please, select one single language."
        language = language[0]
    
    print(f"TA language: {language}")

    if language not in LANGUAGE_MAPPING.keys():
        raise ValueError(
            f"Your chosen language is not supported by this script. Please choose one of the following ones: {', '.join(list(LANGUAGE_MAPPING.keys()))}"
        )
    
    sib200 = load_dataset(dataset, LANGUAGE_MAPPING[language], cache_dir=cache_dir)
    '''sib200 = DatasetDict({
        'train': concatenate_datasets([sib200['train'], sib200['validation']]),
        'test': sib200['test']
        })'''
    
    # add randomness here 
    train_dataset = sib200["train"]
    val_dataset = sib200["validation"]
    test_dataset = sib200["test"]

    if train_size is None and eval_size is None and mode == 'train':   
        return sib200

    elif train_size is None and eval_size is None and mode == 'train':
        return DatasetDict({"train": train_dataset})
    
    elif train_size is None and eval_size is None and mode == 'eval':
        return DatasetDict({"test": test_dataset})
    
    else:
        random.seed(seed)
        if train_size is not None:
            random_indices = random.sample(range(len(train_dataset)), train_size)
            train_dataset = train_dataset.select(random_indices)
            if mode == "train":
                return DatasetDict({"train": train_dataset})             
        
        random.seed(seed)
        if eval_size is not None:
            random_indices = random.sample(range(len(test_dataset)), eval_size)
            test_dataset = test_dataset.select(random_indices)
            if mode == "eval":
                return DatasetDict({"test": test_dataset})

        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

        return dataset


