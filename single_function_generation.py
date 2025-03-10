
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
device = "cuda"
from peft import PeftModel, PeftConfig

from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer
import numpy as np
from utils.set_seed import set_seed
from tqdm import tqdm
import os

set_seed(42)



seq_list = list()


base_model_name_or_path = "/data1/anonymity/Pre_Train_Model/ProtGPT2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path,padding_side = "left") 


task_name_list = ["function_1"]



prefix_name = 'mlpo_model'
for task_name in task_name_list:
    folder_name = f"./{prefix_name}/{task_name}/"


    

    input_p = ['<|endoftext|>']
    inputs = tokenizer(input_p, return_tensors="pt")

    task_name_prefix = {
        "function_0":[0, -1, -1],
        "function_1":[1, -1, -1],
        "process_0":[-1, 0, -1],
        "process_1":[-1, 1, -1],
        "component_0":[-1, -1, 0],
        "component_1":[-1, -1, 1],
        "none":[-1,-1,-1],
    }
    name_prefix = task_name_prefix[task_name]
    max_length = 400



    for subdir in os.listdir(folder_name):
        subdir_path = os.path.join(folder_name, subdir)
        

        if os.path.isdir(subdir_path):
            

            model = PeftModel.from_pretrained(base_model, subdir_path)

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

            result_folder_path = subdir_path.replace(prefix_name,prefix_name+"_result")

            
            os.makedirs(result_folder_path, exist_ok=True)
            result_path = os.path.join(result_folder_path, 'result.txt')

            with open(result_path,"w") as file:
                for seq in seq_list:

                    seq = seq.replace("\n","")
                    seq = seq.rstrip()
                    file.write(f"[{name_prefix}, \"{seq}\"]\n")
                    
            model.to("cpu")

