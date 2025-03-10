
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
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

        parser.add_argument("--lr", type=float, default=5e-5, help="learning_rate")

        parser.add_argument("--dataset_name", type=str, default = "function_0")
            
        parser.add_argument("--wandb", action="store_true")
        
        args = parser.parse_args()

        self.model_name_or_path =  args.model_name_or_path
        self.tokenizer_name_or_path = self.model_name_or_path

        
        
        
        self.dataset_path = os.path.join("dpo_multi_candidate_sequence", args.dataset_name, "mlpo_dataset_multi")   
        
        self.wandb = args.wandb

        name = args.dataset_name.split("_")
        task1_name = name[0]+"_"+name[1]
        task2_name = name[2]+"_"+name[3]
        self.model_path_1 = os.path.join("prefix_tuning_model", task1_name)
        self.model_path_2 = os.path.join("prefix_tuning_model", task2_name)
        
        
        
        
        self.task_type = 'prefix'
        self.dataset_name = args.dataset_name
        
        self.text_column = "Sequence"
        self.max_length = 400
        self.lr = args.lr
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.random_seed = 42
        

        self.output_path = os.path.join("mlpo_multi_model", args.dataset_name)   
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = TrainConfig()


dataset = load_from_disk(config.dataset_path)


original_columns = dataset.column_names

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

tokenizer.pad_token = tokenizer.eos_token



def get_training_model(task1_prefix_path, task2_prefix_path, base_model_path):
    final_model = None
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    prefix_params_list = []
    prefix1_model = PeftModel.from_pretrained(base_model, task1_prefix_path)
    prefix2_model = PeftModel.from_pretrained(base_model, task2_prefix_path)

    for name, param in prefix1_model.named_parameters():
        if 'prompt_encoder' in name:
            prefix_params_list.append(param.data.clone().detach())
    for name, param in prefix2_model.named_parameters():
        if 'prompt_encoder' in name:
            prefix_params_list.append(param.data.clone().detach())
    
        
    final_prefix_params = torch.cat(prefix_params_list, dim=0)
    ref_peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=200)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    ref_model = get_peft_model(base_model, ref_peft_config)
    ref_model.print_trainable_parameters()
    

    for name, param in ref_model.named_parameters():
        if 'prompt_encoder' in name:

            param.data.copy_( final_prefix_params.clone().detach())

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    training_peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=200)
    training_model = get_peft_model(base_model, training_peft_config)
    training_model.print_trainable_parameters()
    
    

    for name, param in training_model.named_parameters():
        if 'prompt_encoder' in name:
            param.data.copy_( final_prefix_params.clone().detach())

    return training_model, ref_model



model, ref_model = get_training_model(config.model_path_1, config.model_path_2, config.model_name_or_path)

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
    
    multi_function = True
)

mlpo_trainer.train()


