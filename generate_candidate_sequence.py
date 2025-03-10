
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import sys
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer
import numpy as np
from utils.set_seed import set_seed
from tqdm import tqdm
import os
import argparse
set_seed(1)

from Bio.PDB import PDBParser


seq_list = list()


import os
base_model_name_or_path = "/data/anonymity/Pre_Train_Model/ProtGPT2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side = "left") 


parser = argparse.ArgumentParser(description="Generate candidate sequence")
parser.add_argument("--model_path", type=str, default='./prefix_tuning_model/', help="Model checkpoint")
args = parser.parse_args()

device = 'cuda'
task_name_list = ["component_0","component_1","process_0","process_1","function_0","function_1"]

for task_name in task_name_list:
    prefix_model_path = os.path.join(args.model_path, task_name)


    input_p = ['<|endoftext|>']
    inputs = tokenizer(input_p, return_tensors="pt")

    task_name_prefix = {
        "function_0":[0, -1, -1],
        "function_1":[1, -1, -1],
        "process_0":[-1, 0, -1],
        "process_1":[-1, 1, -1],
        "component_0":[-1, -1, 0],
        "component_1":[-1, -1, 1],
    }
    name_prefix = task_name_prefix[task_name]
    max_length = 400



    model = PeftModel.from_pretrained(base_model, prefix_model_path)
    model.to(device)
    model.eval()
            

    seq_list = []
    with torch.no_grad():
        gen_seq_num = 50
        batch_size= 50
        seq_list = list()
        try:
            for i in tqdm(range(int(gen_seq_num/batch_size))):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=batch_size, eos_token_id=0)
                seq_res = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                seq_list += seq_res

            remain_seq_num = gen_seq_num-int(gen_seq_num/batch_size)*batch_size
            if remain_seq_num > 0:
                outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=remain_seq_num, eos_token_id=0)
                seq_list += seq_res
        except:
            continue

    result_folder_path = prefix_model_path.replace("prefix_tuning_model","generate_candidate_sequence")

    os.makedirs(result_folder_path, exist_ok=True)
    result_path = os.path.join(result_folder_path, 'result.txt')
    with open(result_path,"w") as file:
        for seq in seq_list:
            seq = seq.split("\n")[0]
            seq = seq.rstrip()
            file.write(f"[{name_prefix}, \"{seq}\"]\n")
            
    model.to("cpu")

