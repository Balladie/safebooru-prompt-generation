import math
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from utils import shuffle_tags


class AnimePromptsDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # eliminate invalid data
        rm_indices = [i for i in tqdm(range(len(self)), desc='Checking validity of data...') if isinstance(self.texts[i], float) and math.isnan(self.texts[i])]
        self.texts = [text for j, text in enumerate(self.texts) if j not in rm_indices]
        print('Nan indices:', rm_indices)
        print('Number of NaN data:', len(rm_indices))
        print('Number of data after remove:', len(self))

    def __getitem__(self, idx):
        text_shuffled = shuffle_tags(self.texts[idx])
        tokenized = self.tokenizer(text_shuffled, truncation=True, padding='max_length', return_tensors='pt')
        return tokenized

    def __len__(self):
        return len(self.texts)


# for debug
if __name__ == '__main__':
    csv_path = 'data/merged_anime_180k_safebooru_2023_filtered_v1.csv'

    texts_data = pd.read_csv(csv_path)
    texts = list(texts_data['safebooru_clean'])

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

    dataset = AnimePromptsDataset(texts, tokenizer)