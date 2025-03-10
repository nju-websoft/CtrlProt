
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
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
from tqdm import tqdm
from utils.log_helper import logger_init
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk
import logging
import random
from utils.set_seed import set_seed
import time
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb
import wandb
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
import json
from trl import DPOTrainer, ORPOTrainer
from mlpo_trainer import MLPOTrainer


seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class TrainConfig:
    def __init__(self):   
        parser = argparse.ArgumentParser(description="Train prefix_tuning_prot")
        parser.add_argument("--model_name_or_path", type=str, default='/data1/anonymity/Pre_Train_Model/ProtGPT2/', help="Model checkpoint")

        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

        parser.add_argument("--lr", type=float, default=1e-5, help="learning_rate")
        parser.add_argument("--dataset_path", type=str, default = "./dataset/function/0.tsv")
        parser.add_argument("--dataset_name", type=str, default = "function_0")
        
        parser.add_argument("--model_path", type=str, default = "/data1/anonymity/CtrlProt/best_prefix_tuning_model/function_0")
        
        parser.add_argument("--output_path", type=str, default = "./saved_model/")
        
        
        parser.add_argument("--wandb", action="store_true")
        
        args = parser.parse_args()

        self.model_name_or_path =  args.model_name_or_path
        self.tokenizer_name_or_path = self.model_name_or_path
            
        self.dataset_path = args.dataset_path    
        
        self.wandb = args.wandb

            
        self.model_path = args.model_path
        self.num_virtual_tokens = 100

        self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=self.num_virtual_tokens)
        
        self.task_type = 'prefix'
        self.dataset_name = args.dataset_name
        
        self.text_column = "Sequence"
        self.max_length = 400
        self.lr = args.lr
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.random_seed = 42
        
        self.output_path = self.model_path.replace("prefix_tuning_model","mlpo_model")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TrainConfig()


dataset = load_from_disk(config.dataset_path)


original_columns = dataset.column_names


tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
model = PeftModel.from_pretrained(model, config.model_path)


model.print_trainable_parameters()

ref_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
ref_model = PeftModel.from_pretrained(ref_model, config.model_path)


output_path = config.output_path

gradient_accumulation_steps = 1
if config.batch_size == 16:
    gradient_accumulation_steps = 2
if config.batch_size == 8:
    gradient_accumulation_steps = 4


training_args = TrainingArguments(
    per_device_train_batch_size = config.batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,

    num_train_epochs= config.num_epochs,
    learning_rate=config.lr,
    
    lr_scheduler_type="cosine",
   
    logging_steps=10,

    output_dir=config.output_path,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to='wandb' if config.wandb else 'none',
    
    
    
)


mlpo_trainer = MLPOTrainer(
    model = model,
    ref_model = ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,

    beta=0.1,
    max_prompt_length=1,
    max_length=400,

    
    
    alpha = 0.05,

)

mlpo_trainer.train()


