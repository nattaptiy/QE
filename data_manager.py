from os import path
from csv import QUOTE_NONE
import torch
import pandas as pd
from transformers import AutoTokenizer
from config.mlqe import BATCH_SIZE, SOURCE, TARGET, LEVEL, DEVICE
from data.data_manager import DataManager as DataManagerMeta
from torch.utils.data import TensorDataset, DataLoader


class DataManager(DataManagerMeta):
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 batch_size=BATCH_SIZE,
                 source=SOURCE,
                 target=TARGET,
                 level=LEVEL,
                 device=DEVICE):
        self.batch_size = batch_size
        self.source_label = source
        self.target_label = target
        self.level_label = level
        self.device = device
        self.to_torch_long = True
        self.tokenizer = tokenizer
        self.dataset_item_order = ["input_ids", "attention_mask", "labels"]

    def generate_example(self, **kwargs):
        pass

    def generate_tensor_dataset_from_path(self, filepath):
        sequence_pairs = list()
        levels = list()
        df = pd.read_csv(filepath, delimiter="\t", quoting=QUOTE_NONE)
        for sentence1, sentence2, level in zip(df[self.source_label],
                                               df[self.target_label],
                                               df[self.level_label]):
            sentence1 = sentence1.strip()
            sentence2 = sentence2.strip()
            #level = float(level.strip())
            levels.append(level)

            sequence_pairs.append([sentence1, sentence2])
        result = self.tokenizer(sequence_pairs, padding=True)
        result["labels"] = levels # this is due to the transformers Classification class
        tensor_list = list()
        for key in self.dataset_item_order:
            if key != "labels":
                tensor_list.append(torch.tensor(result[key], dtype=torch.long))
            else:
                tensor_list.append(torch.tensor(result[key], dtype=torch.float))
        result = TensorDataset(*tensor_list)
        return result

    def load_file(self, filepath, manage_cache=True, cache_ext=".cache", do_eval=False, clear_cache=False):
        if manage_cache:
            cache_filepath = filepath + cache_ext
            if path.isfile(cache_filepath) and not clear_cache:  # if cache file is exists, then load it
                dataset = self.generate_tensor_dataset_from_path(filepath)
            else:  # otherwise, save cache
                dataset = self.generate_tensor_dataset_from_path(filepath)
                torch.save(dataset, cache_filepath)
        else:
            dataset = self.generate_tensor_dataset_from_path(filepath)
        if do_eval:
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        else:
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader

    def print_batch(self, batch, print_all=False):
        print()
        print()
        for key, value in enumerate(batch):
            key = self.dataset_item_order[key]
            if key == "input_ids":
                print(key, self.tokenizer.decode(value[0, :].tolist()))
            if key in ["input_ids", "attention_mask"]:
                print(key, value[0, :].tolist())
            else:
                print(key, value[0].item())

    def fix_batch(self, batch, device="cpu"):
        inputs = dict()
        for idx, key in enumerate(self.dataset_item_order):
            inputs[key] = batch[idx]

        keys = list(inputs.keys())
        for key in keys:
            if key not in ["input_ids", "attention_mask", "labels"]:
                del inputs[key]

        # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": 0, "labels": batch[3]}
        return inputs
