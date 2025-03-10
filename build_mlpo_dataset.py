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
            if label[0]!=-1:
                new_label_score = new_label_score[0]
            elif label[1]!=-1:
                new_label_score = new_label_score[1]
            else:
                new_label_score = new_label_score[2]
            if rosetta_energy == -1000:
                continue
            samples.append({
                "label": label,
                "sequence": sequence,
                "new_label_score": new_label_score,
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
    

def filter_random_pairs(samples, num_pairs=5000):
    filtered_pairs = []
    num_samples = len(samples)
    



    while len(filtered_pairs) < num_pairs:

        idx1, idx2 = random.sample(range(num_samples), 2)
        item1 = samples[idx1]
        item2 = samples[idx2]

        if (item1['new_label_score'] > item2['new_label_score'] and 
            item1['rosetta_energy'] > item2['rosetta_energy']):
            filtered_pairs.append((item1, item2))
            

    return filtered_pairs



def calculate_samples(samples):
    random.seed(42)
    dataset_energy = []
    
    preferred_samples = []
    high_rosetta_low_label_score = []
    low_rosetta_high_label_score = []
    low_samples = []
    
    rosetta_energy_min = min([sample["rosetta_energy"] for sample in samples])
    rosetta_energy_max = max([sample["rosetta_energy"] for sample in samples])
        
    for sample in samples:
        sample["rosetta_energy"] =(sample["rosetta_energy"] - rosetta_energy_min)/(rosetta_energy_max - rosetta_energy_min)


    epsilon = 1e-5

    rosetta_energies = np.array([sample["rosetta_energy"] for sample in samples])

    rosetta_energies = np.clip(rosetta_energies, epsilon, 1 - epsilon)
    a, b, loc, scale = beta.fit(rosetta_energies, floc=0, fscale=1)
    

    for sample in samples:
        sample["rosetta_energy_cdf"] = beta.cdf(sample["rosetta_energy"], a, b, loc, scale)
    


    label_scores = np.array([sample["new_label_score"] for sample in samples])
    label_scores = np.clip(label_scores, epsilon, 1 - epsilon)
    a, b, loc, scale = beta.fit(label_scores, floc=0, fscale=1)
    

    for sample in samples:
        sample["label_score_cdf"] = beta.cdf(sample["new_label_score"], a, b, loc, scale)

    
    return samples
           

def plot_regularization_distribution(regularization_term, path="regularization_term_distribution.png"):
    plt.figure()
    plt.hist(regularization_term, bins=30, alpha=0.75, color='blue')
    plt.xlabel('Regularization Term Value')
    plt.ylabel('Frequency')
    plt.title('Regularization Term Distribution')
    plt.grid(True)
    plt.savefig(path)
    plt.show()

    

def create_mlpo_dataset(samples, tokenizer):
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
    
    
    

    pair_list = filter_random_pairs(samples, num_pairs=5000)
    random.shuffle(pair_list)
    
    preferred_samples_sequence = [prefer['sequence'] for prefer, reject in pair_list]
    rejected_samples_sequence = [reject['sequence'] for prefer, reject in pair_list]
    
    for prefer, reject in pair_list:
        candidate_preferred_samples.append(prefer)
        candidate_rejected_samples.append(reject)
    
    chosen_score = []
    for sample in candidate_preferred_samples:
        function_score = math.pow(2, sample["new_label_score"]) - 1
        rosetta_score = math.pow(2, sample["rosetta_energy"]) - 1
        chosen_score.append([function_score, rosetta_score])
    
    rejected_score = []
    for sample in candidate_rejected_samples:
        function_score = math.pow(2, sample["new_label_score"]) - 1
        rosetta_score = math.pow(2, sample["rosetta_energy"]) - 1
        rejected_score.append([function_score, rosetta_score])
    
    
    chosen_weight = []
    for sample in candidate_preferred_samples:
        sum = sample["label_score_cdf"] + sample["rosetta_energy_cdf"]
        chosen_weight.append([sample["label_score_cdf"], sample["rosetta_energy_cdf"] ])
    
    rejected_weight = []
    for sample in candidate_rejected_samples:
        sum = sample["label_score_cdf"] + sample["rosetta_energy_cdf"]
        rejected_weight.append([sample["label_score_cdf"], sample["rosetta_energy_cdf"]])
    
    
    regularization_term = []
    
    for i in range(len(candidate_preferred_samples)):
        chosen_term = chosen_weight[i][0] * chosen_score[i][0] + chosen_weight[i][1] * chosen_score[i][1]
        rejected_term = rejected_weight[i][0] * rejected_score[i][0] + rejected_weight[i][1] * rejected_score[i][1]
        regularization_term.append(chosen_term - rejected_term)
    
    
    


    
    
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
    
    threshold_dict = {"function_0":[0.85, 1], "function_1":[0.85, 1], "process_0":[0.85, 1], "process_1":[0.85, 1], "component_0":[0.85, 1], "component_1":[0.85, 1]}
    
    for task_name in threshold_dict.keys():
        temp_file_path = os.path.join(file_path, task_name, "result.txt")
        samples = read_protein_data(temp_file_path)
        samples = calculate_samples(samples)
        mlpo_dataset = create_mlpo_dataset(samples, tokenizer)
        output_path = temp_file_path.replace("result.txt", "mlpo_dataset")
        mlpo_dataset.save_to_disk(output_path)
        
if __name__ == "__main__":
    main()
