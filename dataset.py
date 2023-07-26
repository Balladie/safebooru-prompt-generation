import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from utils import shuffle_tags


class AnimePromptsDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __getitem__(self, idx):
        text_shuffled = shuffle_tags(self.texts[idx])
        tokenized = self.tokenizer(text_shuffled, truncation=True, padding='max_length', return_tensors='pt')
        return tokenized

    def __len__(self):
        return len(self.texts)