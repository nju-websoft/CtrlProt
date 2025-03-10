
import os

import sys
sys.path.append('./')
import numpy as np
import argparse
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
from utils.log_helper import logger_init
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import logging
import random
from utils.set_seed import set_seed
import time
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler



class TrainConfig:
    def __init__(self):
        
        
        parser = argparse.ArgumentParser(description="Train prefix_tuning_prot")
        parser.add_argument("--model_name_or_path", type=str, default='/data/anonymity/Pre_Train_Model/ProtGPT2/', help="Model checkpoint")

        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

        
        parser.add_argument("--dataset_path", type=str, default = "./dataset/function/0.tsv")
        parser.add_argument("--dataset_name", type=str, default = "function_0")
        
        parser.add_argument("--lr",type=float, default=5e-5)
        parser.add_argument("--output_path", type=str, default = "./saved_model/")
        args = parser.parse_args()


        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_name_or_path =  args.model_name_or_path
        self.tokenizer_name_or_path = self.model_name_or_path
        
        
        self.dataset_path = args.dataset_path
        
        

        
        self.num_virtual_tokens = 100

        self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=self.num_virtual_tokens)
        
        self.task_type = 'prefix'
        self.dataset_name = args.dataset_name
        
        self.text_column = "Sequence"
        self.max_length = 404
        self.lr = 1e-4
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.random_seed = 42

        self.model_save_dir = os.path.join(self.project_dir, 'saved_dir', f'{self.dataset_name}', f'{self.task_type}')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs', f'{self.dataset_name}', f'{self.task_type}')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)


def train(config):
    
    
    scaler = GradScaler()


    
    data_files = {"train": config.dataset_path}
    
    dataset = load_dataset('csv', data_files=config.dataset_path, split = 'train')
    dataset = dataset.train_test_split(test_size=0.1)
    logging.info('check the info about dataset')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path) 


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    def preprocess_function(examples):
        batch_size = len(examples["Entry\tSequence"])
        print(batch_size)
        inputs = [x.split("\t")[-1] for x in examples["Entry\tSequence"]]
        model_inputs = tokenizer(inputs)
        labels = model_inputs

        for i in range(batch_size):
            sample_input_ids = [tokenizer.eos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
            label_input_ids = [tokenizer.eos_token_id] + labels["input_ids"][i] + [tokenizer.eos_token_id]
            labels["input_ids"][i] = label_input_ids
            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])


        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id] * (
                config.max_length - len(sample_input_ids)
            )
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] +  [0] * (config.max_length - len(sample_input_ids)) 
            labels["input_ids"][i] = label_input_ids + [0] * (config.max_length - len(sample_input_ids))
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:config.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:config.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:config.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs



    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    logging.info(processed_datasets)



    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['test']

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=config.batch_size, pin_memory=True)


    if config.task_type == 'prefix':
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        model = get_peft_model(model, config.peft_config)
        model.print_trainable_parameters()
    elif config.task_type == 'finetune':
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)



    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs), lr_end=1e-7, power=3
    )


    model = model.to(config.device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    time_start = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            global_iter_num = epoch * len(train_dataloader) + step + 1
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            total_loss += loss.detach().float()

            
            scaler.step(optimizer)
            scaler.update()
            
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader, ncols=50)):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            with torch.no_grad():
                with autocast():
                    outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        if epoch %10 ==0:
            peft_model_id = config.model_save_dir + f"/E{epoch}_VT{config.num_virtual_tokens}_eval_loss{np.around(eval_epoch_loss.cpu(), 5)}_{np.around(eval_ppl.cpu(), 5)}"
            model.save_pretrained(peft_model_id)
    time_end = time.time()
    time_sum = time_end - time_start
    config.writer.add_scalar('training time', time_sum, 0)
    logging.info(f'The sum time of training {time_sum}')


    if config.task_type == 'prefix':
        peft_model_id = config.model_save_dir + f"/E{epoch}_VT{config.num_virtual_tokens}_eval_loss{np.around(eval_epoch_loss.cpu(), 5)}_{np.around(eval_ppl.cpu(), 5)}"
        model.save_pretrained(peft_model_id)



if __name__ == '__main__':
    train_config = TrainConfig()
    set_seed(train_config.random_seed)
    train(train_config)
    
