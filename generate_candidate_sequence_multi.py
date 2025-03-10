
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType, PeftModel
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer
import numpy as np
from utils.set_seed import set_seed
from tqdm import tqdm
import os

set_seed(42)



def get_generation_model(task1_prefix_path, task2_prefix_path, base_model, mode):
    final_model = None

    prefix_params_list = []
    prefix1_model = PeftModel.from_pretrained(base_model, task1_prefix_path)
    prefix2_model = PeftModel.from_pretrained(base_model, task2_prefix_path)


    for name, param in prefix1_model.named_parameters():
        if 'prompt_encoder' in name:
            prefix_params_list.append(param.data.clone().detach())
    for name, param in prefix2_model.named_parameters():
        if 'prompt_encoder' in name:
            prefix_params_list.append(param.data.clone().detach())
    
        
    if mode == "average":

        final_prefix_params = torch.mean(torch.stack(prefix_params_list), dim=0)
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=100)
        final_model = get_peft_model(base_model, peft_config)
        final_model.print_trainable_parameters()
        
    elif mode == "concat":

        final_prefix_params = torch.cat(prefix_params_list, dim=0)
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=200)
        final_model = get_peft_model(base_model, peft_config)
        final_model.print_trainable_parameters()
    
    

    for name, param in final_model.named_parameters():
        if 'prompt_encoder' in name:

            param.data.copy_( final_prefix_params.clone().detach())

    return final_model

seq_list = list()


device = "cuda"
base_model_name_or_path = "/data1/anonymity/Pre_Train_Model/ProtGPT2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path) 

task_combination_list = ["process_component", "function_process","function_component"]

mode = "concat"
output_folder = f"/data1/anonymity/CtrlProt/generate_candidate_sequence_multi/"

task_name_list = [("function_0","component_1"),("function_0","process_0")]

    

for (task1, task2) in task_name_list:
    folder_path = os.path.join(output_folder,  task1+"_"+task2)
    
    

    task_name_prefix = {"function":-1, "process":-1, "component":-1}
    x,y = task1.split("_")
    task_name_prefix[x] = int(y)
    x,y = task2.split("_")
    task_name_prefix[x] = int(y)
    
    task1_prefix_path = f"/data1/anonymity/CtrlProt/prefix_tuning_model/{task1}"
    task2_prefix_path = f"/data1/anonymity/CtrlProt/prefix_tuning_model/{task2}"
    generation_model = get_generation_model(task1_prefix_path, task2_prefix_path, base_model, mode = mode).to(device)
    generation_model.eval()
    input_p = ['<|endoftext|>']
    inputs = tokenizer(input_p, return_tensors="pt")

    max_length = 400

    
    seq_list = []
    with torch.no_grad():
        gen_seq_num = 5000
        batch_size= 50
        seq_list = list()
        try:
            for i in tqdm(range(int(gen_seq_num/batch_size))):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = generation_model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=batch_size, eos_token_id=0)
                seq_res = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                seq_list += seq_res

            remain_seq_num = gen_seq_num-int(gen_seq_num/batch_size)*batch_size
            if remain_seq_num > 0:
                outputs = generation_model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length, do_sample=True, top_k=500, repetition_penalty=1.2, num_return_sequences=remain_seq_num, eos_token_id=0)
                seq_list += seq_res
        except:
            continue

    result_folder_path = os.path.join(output_folder, task1 + "_" + task2)
    
    prefix = [task_name_prefix["function"],task_name_prefix["process"],task_name_prefix["component"]]
    os.makedirs(result_folder_path, exist_ok=True)
    result_path = os.path.join(result_folder_path, 'result.txt')
    with open(result_path,"w") as file:
        for seq in seq_list:
            seq = seq.split("\n")[0]
            seq = seq.rstrip()
            file.write(f"[{prefix}, \"{seq}\"]\n")
    generation_model.to("cpu")

