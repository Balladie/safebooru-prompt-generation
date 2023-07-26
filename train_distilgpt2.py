import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from dataset import AnimePromptsDataset
from utils import parse_args, get_logger


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger = get_logger()
    logger.warning(f"device: {device}, n_gpu: {n_gpu}")

    if args.seed:
        set_seed(args.seed)

    # read data from .csv file
    texts_data = pd.read_csv(args.csv_path).dropna()
    texts = list(texts_data['safebooru_clean'])

    # data collator
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # prepare dataset
    train_texts, test_texts = train_test_split(texts, test_size=0.05)
    train_dataset = AnimePromptsDataset(train_texts, tokenizer)
    test_dataset = AnimePromptsDataset(test_texts, tokenizer)

    # load model
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')

    # training configs
    training_args = TrainingArguments(
        output_dir="runs", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=3, # number of training epochs
        per_device_train_batch_size=4, # batch size for training
        per_device_eval_batch_size=4,  # batch size for evaluation
        save_strategy='steps',
        save_steps=2000, # after # steps model is saved 
        warmup_steps=1000,# number of warmup steps for learning rate scheduler
        learning_rate=1e-5,
        evaluation_strategy='steps',
        eval_steps=2000,
        #prediction_loss_only=True,
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()