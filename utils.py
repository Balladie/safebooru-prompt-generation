import logging
import time
import os
import sys
import argparse
import random
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--csv_path', type=str, default='data/anime_prompts_180k_filtered_v1.csv')

    return parser.parse_args()


def parse_args_test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)

    return parser.parse_args()


def get_logger():
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join('logs', f'{time.time()}.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def shuffle_tags(text):
    tokens = text.split(', ')
    
    copy_except_first = tokens[1:]
    random.shuffle(copy_except_first)
    tokens[1:] = copy_except_first

    text_shuffled = ', '.join(tokens)

    return text_shuffled


def merge_two_csv(csv_path_1, csv_path_2):
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    df_merged = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    df_merged.to_csv('data/merged_anime_180k_safebooru_2023_filtered_v1.csv', index=False)


if __name__ == '__main__':
    csv_path_1 = 'data/anime_prompts_180k_filtered_v1.csv'
    csv_path_2 = 'data/safebooru_prompts_2023_filtered_v1.csv'

    merge_two_csv(csv_path_1, csv_path_2)