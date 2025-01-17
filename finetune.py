import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, TrainingArguments, Trainer, LLaMAForCausalLM, IntervalStrategy, LLaMATokenizer
import json
import argparse
from minimal_llama.utils import load_yaml, load_jsonl, freeze_bottom_causal_layers
import wandb
from datasets import load_dataset, DownloadMode, load_from_disk
from torch.utils.data import DataLoader

from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed

import numpy as np


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones_like(inputs["input_ids"]),
            labels=inputs["input_ids"],  # HF model does the slicing for us
        ).loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(config):
    try:
        tokenizer = LLaMATokenizer.from_pretrained(
                config['tokenizer_path'],
                padding_side="left",
                )
    except ValueError:
        tokenizer = LLaMATokenizer.from_pretrained(
                config['tokenizer_path'],
                padding_side="left",
                use_fast=False,
                )    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    training_args = TrainingArguments(**config["train_args"])
    model = LLaMAForCausalLM.from_pretrained(config["model_path"]).cuda()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("Setup Data")
    dataset = load_from_disk(config["dataset_path"])
    print("Len data: ", len(dataset))

    # take top 100 examples
    dataset = dataset.select(range(100))

    train_size = int(0.94 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])


    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=data_collator).train()

    model.save_pretrained(config["save_dir"])

    if torch.distributed.get_rank() == 0:
        if os.environ.get('DEEPSPEED_ZERO_STAGE', '0') != '3':
            EOS_ID = tokenizer("<|endoftext|>")["input_ids"][0]
            data = []
            for i in range(16):
                prompt = val_dataset[i]["text"]
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].view(1, -1).cuda()
                attention_mask = inputs["attention_mask"].view(1, -1).cuda()
                sample_outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, max_length=1024)
                response = tokenizer.batch_decode(sample_outputs)[0].split("<|endoftext|>")[0][len(prompt):]
                data.append([prompt, response])
            cols = ["prompt", "response"]
            wandb.log({"samples": wandb.Table(columns=cols, data=data)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--ds_config_path", type=str)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config_path)
    config["train_args"]["deepspeed"] = args.ds_config_path

    train(config)