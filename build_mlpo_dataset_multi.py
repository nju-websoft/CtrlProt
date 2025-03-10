import random
import json
import argparse
from datasets import Dataset
import random
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import math
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import beta
import random


def read_protein_data(file_path):
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            label, sequence, new_label_score, plddt, rosetta_energy = json.loads(line.rstrip())
            score = []
            if label[0]!=-1:
               score.append(new_label_score[0])
            if label[1]!=-1:
                score.append(new_label_score[1])
            if label[2]!=-1:
                score.append(new_label_score[2])
                
            if rosetta_energy == -1000:
                continue
            samples.append({
                "label": label,
                "sequence": sequence,
                "new_label_score_1": score[0],
                "new_label_score_2": score[1],
                "plddt": float(plddt),
                "rosetta_energy": float(-rosetta_energy)
            })   
    return samples



def plot_distribution(data, a, b, title, filename):
    x = np.linspace(0, 1, 100)
    
    plt.figure()
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Original Data')
    plt.plot(x, beta.pdf(x, a, b), 'r-', lw=2, label='Fitted Beta Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    

def filter_samples(samples, label1_score_threshold, label2_score_threshold, rosetta_threshold):
    random.seed(42)
    dataset_energy = []
    
    preferred_samples = []
    high_rosetta_low_label_score = []
    low_rosetta_high_label_score = []
    low_samples = []
    
    rosetta_energy_min = min([sample["rosetta_energy"] for sample in samples])
    rosetta_energy_max = max([sample["rosetta_energy"] for sample in samples])
        
    for sample in samples:
        if sample["new_label_score_1"] >= label1_score_threshold and sample["new_label_score_2"] >= label2_score_threshold and sample["rosetta_energy"] >= rosetta_threshold:
            preferred_samples.append(sample)
        else:
            low_samples.append(sample)

    num1=0
    num2=0
    num3=0
    for sample in samples:
        if sample["new_label_score_1"] >= label1_score_threshold:
            num1+=1
            
        if sample["new_label_score_2"] >= label2_score_threshold:
            num2+=1
        
        if sample["rosetta_energy"] >= rosetta_threshold:
            num3+=1
            
    print(f"new_label_score1 preferred num: {num1}")
    print(f"new_label_score2 preferred num: {num2}")
    print(f"rosetta energy preferred num: {num3}")
    
    for sample in samples:
        sample["rosetta_energy"] =(sample["rosetta_energy"] - rosetta_energy_min)/(rosetta_energy_max - rosetta_energy_min)
    
    epsilon = 1e-5

    rosetta_energies = np.array([sample["rosetta_energy"] for sample in samples])

    rosetta_energies = np.clip(rosetta_energies, epsilon, 1 - epsilon)
    a, b, loc, scale = beta.fit(rosetta_energies, floc=0, fscale=1)

    for sample in samples:
        sample["rosetta_energy_cdf"] = beta.cdf(sample["rosetta_energy"], a, b, loc, scale)
    plot_distribution(rosetta_energies, a, b, "Rosetta Energy Distribution", "rosetta_energy_distribution.png")
    

    label_scores = np.array([sample["new_label_score_1"] for sample in samples])
    label_scores = np.clip(label_scores, epsilon, 1 - epsilon)
    a, b, loc, scale = beta.fit(label_scores, floc=0, fscale=1)

    for sample in samples:
        sample["label_score_cdf_1"] = beta.cdf(sample["new_label_score_1"], a, b, loc, scale)
    plot_distribution(label_scores, a, b, "Label Score Distribution", "label_score_distribution.png")


    label_scores = np.array([sample["new_label_score_2"] for sample in samples])
    label_scores = np.clip(label_scores, epsilon, 1 - epsilon)
    a, b, loc, scale = beta.fit(label_scores, floc=0, fscale=1)
    for sample in samples:
        sample["label_score_cdf_2"] = beta.cdf(sample["new_label_score_2"], a, b, loc, scale)
    plot_distribution(label_scores, a, b, "Label Score Distribution", "label_score_distribution.png")
    
    
    
    return preferred_samples, low_samples
           

def plot_regularization_distribution(regularization_term, path="regularization_term_distribution.png"):
    plt.figure()
    plt.hist(regularization_term, bins=30, alpha=0.75, color='blue')
    plt.xlabel('Regularization Term Value')
    plt.ylabel('Frequency')
    plt.title('Regularization Term Distribution')
    plt.grid(True)
    plt.savefig(path)
    plt.show()

    

def create_mlpo_dataset(preferred_samples, rejected_samples, tokenizer):
    random.seed(42)
    candidate_preferred_samples =[]
    candidate_rejected_samples =[]
    preferred_score =[]
    preferred_rank =[]
    rejected_score =[]
    rejected_rank =[]

    lambda_weight_1 = []
    lambda_weight_2 = []
    
    num_samples = 5000
    
    
    
    candidate_preferred_samples = random.choices(preferred_samples, k = num_samples)
    candidate_rejected_samples = random.choices(rejected_samples, k= num_samples)
    
    preferred_samples_sequence = [sample['sequence'] for sample in candidate_preferred_samples]
    rejected_samples_sequence = [sample['sequence'] for sample in candidate_rejected_samples]
    
    chosen_score = []
    for sample in candidate_preferred_samples:
        function_score1 = math.pow(2, sample["new_label_score_1"]) - 1
        function_score2 = math.pow(2, sample["new_label_score_2"]) - 1
        rosetta_score = math.pow(2, sample["rosetta_energy"]) - 1
        chosen_score.append([function_score1, function_score2, rosetta_score])
    
    rejected_score = []
    for sample in candidate_rejected_samples:
        function_score1 = math.pow(2, sample["new_label_score_1"]) - 1
        function_score2 = math.pow(2, sample["new_label_score_2"]) - 1
        rosetta_score = math.pow(2, sample["rosetta_energy"]) - 1
        rejected_score.append([function_score1, function_score2, rosetta_score])
    
    

    

    chosen_weight = []
    for sample in candidate_preferred_samples: 
        chosen_weight.append([sample["label_score_cdf_1"], sample["label_score_cdf_2"], sample["rosetta_energy_cdf"] ])
    
    rejected_weight = []
    for sample in candidate_rejected_samples:
        rejected_weight.append([sample["label_score_cdf_1"], sample["label_score_cdf_2"], sample["rosetta_energy_cdf"]])
    
    
    regularization_term = []
    
    for i in range(len(candidate_preferred_samples)):
        chosen_term = (chosen_weight[i][0] * chosen_score[i][0] + chosen_weight[i][1] * chosen_score[i][1])/2+ chosen_weight[i][2] * chosen_score[i][2]
        rejected_term = (rejected_weight[i][0] * rejected_score[i][0] + rejected_weight[i][1] * rejected_score[i][1])/2 + rejected_weight[i][2] * rejected_score[i][2]
        regularization_term.append(chosen_term - rejected_term)
    
    

    plot_regularization_distribution(regularization_term)
    
    
    mlpo_dataset = {

        "prompt": ['']*num_samples,
        "chosen": preferred_samples_sequence,
        "rejected": rejected_samples_sequence,
        
        "chosen_score": chosen_score,
        "chosen_weight": chosen_weight,
        "rejected_score": rejected_score,
        "rejected_weight": rejected_weight,
        
        "regularization_term": regularization_term
        }
    
    
    mlpo_dataset = Dataset.from_dict(mlpo_dataset)
    
    
    
    return mlpo_dataset
    
    
    
    
    
    
 

def save_dpo_dataset(dpo_dataset, file_path):
    with open(file_path, 'w') as file:
        json.dump(dpo_dataset.to_dict(), file)


def load_dpo_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def main():
    
    parser = argparse.ArgumentParser(description="build_mlpo_dataset")
    parser.add_argument("--model_path", type=str, default = "/data1/anonymity/Pre_Train_Model/ProtGPT2")
    parser.add_argument("--dataset_path", type=str, default = "./saved_model/")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:   
        tokenizer.pad_token_id = tokenizer.eos_token_id
     
    file_path = args.dataset_path
    

    threshold_dict = {"function_0_component_1":[0.75, 0.85, 0.7], "function_0_process_0":[0.8, 0.85, 1], 
                      "function_1_process_1":[0.85, 0.85,1], "process_0_component_1":[0.8, 0.9,1], 
                      "process_1_component_0":[0.85,0.85, 1]}
    
    
    
    for task_name in threshold_dict.keys():
        
        temp_file_path = os.path.join(file_path, task_name, "result.txt")

        label1_score_threshold = threshold_dict[task_name][0]
        label2_score_threshold = threshold_dict[task_name][1]
        rosetta_threshold = threshold_dict[task_name][2]
        
        print(task_name)
        print(f"using label threshold1 {label1_score_threshold}")
        print(f"using label threshold2 {label2_score_threshold}")
        print(f"using rosetta threshold {rosetta_threshold}")

        samples = read_protein_data(temp_file_path)
        
        preferred_samples, low_samples = filter_samples(samples, label1_score_threshold, label2_score_threshold, rosetta_threshold)
        

        mlpo_dataset = create_mlpo_dataset(preferred_samples, low_samples, tokenizer)
        output_path = temp_file_path.replace("result.txt", "mlpo_dataset_multi")
        mlpo_dataset.save_to_disk(output_path)
        
if __name__ == "__main__":
    main()
