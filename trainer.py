import os
import argparse
import torch
import wandb
import random

from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset

from modules.config import Config

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
random.seed(42)

def tokenize_text(examples):
    global tokenizer
    global config
    inputs = tokenizer(
        examples['text'], 
        padding="max_length",
        truncation=True,
        max_length=config.max_seq_length,
        return_tensors="pt"
    )
    return inputs

# if we are on windows, we need to check it, and set the torch backend to gloo
if os.name == 'nt':
    try:    
        torch.distributed.init_process_group(backend="gloo")
    except:
        pass

accelerator = Accelerator()
device = accelerator.device if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to the config file')
parser.add_argument('-w', '--wandb', action='store_true', help='Use wandb for logging')
parser.add_argument('-r', '--resume', type=str, default=None, help='Path to the checkpoint to resume training')

args = parser.parse_args()

config = Config(args.config_path)

# Initialize the tokenizer
if args.resume is not None:
    tokenizer = AutoTokenizer.from_pretrained(args.resume, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True)

if __name__ == '__main__':
    # Initialize the model
    if args.resume is not None:
        model = AutoModelForMaskedLM.from_pretrained(args.resume)
    else:
        model = AutoModelForMaskedLM.from_pretrained(config.model)
    
    model.to(device)
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print(f"Model initialized with {model.num_parameters()} parameters.")
        print(f"Tokenizer initialized with {len(tokenizer)} tokens.")

    # Load the dataset
    train_dataset = load_dataset(config.train_dataset, **config.train_dataset_kwargs)
    eval_dataset = load_dataset(config.eval_dataset, **config.eval_dataset_kwargs)
    
    # Apply tokenization on the fly
    train_dataset.set_transform(tokenize_text)
    eval_dataset.set_transform(tokenize_text)

    # define the function inplace
    # train_dataset.set_transform( lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=config.max_seq_length, return_tensors="pt") )

    if accelerator.is_main_process:    
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    if config.num_epochs is None:
        num_epochs = config.max_steps * config.batch_size / len(train_dataset)
    else:
        num_epochs = config.num_epochs

    if accelerator.is_main_process:
        print("--- Hyperparameters ---")
        for key in config._jsonData.keys():
            print(f"{key}: {config._jsonData[key]}")
        print("-----------------------")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_dir=config.output_dir,
        save_strategy="epoch" if config.num_epochs is not None else "steps",
        save_steps=config.save_steps,
        eval_strategy="epoch" if config.num_epochs is not None else "steps",
        eval_steps=config.save_steps,
        seed=4242,
        bf16=True,
        report_to="wandb" if args.wandb else "none",
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=True,
        dataloader_num_workers=config.num_workers,
        warmup_steps=config.warmup_steps,
        # include_tokens_per_second=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=config.mlm_probability)

    trainer = Trainer(
        model=model,        
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[]
    )

    if args.wandb and accelerator.is_main_process:
        wandb.init(project=config.wandb["project"], name=config.wandb["name"], tags=config.wandb["tags"])
        wandb.config.update(config._jsonData)
        wandb.watch(model)

    model.config.use_cache = False # mute warnings


    model, train_dataset, eval_dataset = accelerator.prepare(model, train_dataset, eval_dataset)
 
    trainer.train()