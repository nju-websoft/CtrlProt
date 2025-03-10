import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
from torch import cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
import torch
from PIL import Image
import torch
import argparse
import statistics
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

from Bio.PDB import PDBParser
import numpy as np, pickle
from pyrosetta import *
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, EsmForMaskedLM, EsmModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
parser = PDBParser(QUIET=True)


init()

scorefxn =  get_fa_scorefxn()   

torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda:0"
def evaluate_structure_score(input_structure_path, training_structure_path):

    def load_and_average_embedding(file_path):
        data = np.load(file_path, allow_pickle=True)
        emb = data["mpnn_emb"]
        return torch.mean(emb, axis=0)

    training_set = []
    for root, _, files in os.walk(training_structure_path):
        for file in files:
            if file.endswith(".pyd"):
                file_path = os.path.join(root, file)
                avg_embedding = load_and_average_embedding(file_path)
                training_set.append(avg_embedding)
    training_set = torch.stack(training_set)
    
    scores = {}

    for root, _, files in os.walk(input_structure_path):
        for file in tqdm(files):
            if file.endswith(".pyd"):
                file_path = os.path.join(root, file)
                avg_embedding = load_and_average_embedding(file_path)
                avg_embedding_tensor = torch.tensor(avg_embedding)
                
                cosine_similarities = torch.nn.functional.cosine_similarity(avg_embedding_tensor, training_set, dim=1)
            

                top_100_mean = torch.mean(torch.topk(cosine_similarities, 1000).values)
                scores[int(file.split(".")[0])] = top_100_mean.item()

    x = 1

    score_values = list(scores.values())


    mean_score = statistics.mean(score_values)
    max_score = max(score_values)
    min_score = min(score_values)
    std_dev_score = statistics.stdev(score_values)
    median_score = statistics.median(score_values)

    print(f"Mean: {mean_score}")
    print(f"Max: {max_score}")
    print(f"Min: {min_score}")
    print(f"Standard Deviation: {std_dev_score}")
    print(f"Median: {median_score}")
    return scores



def get_rosetta_energy(relax_path, length):
    energy_dict = {}
    pdb_list = os.listdir(relax_path)
    for file_path in tqdm(pdb_list):
        try:
            index = int(file_path.split(".")[0])
            pdb_file_path = os.path.join(relax_path, file_path)
            pose = pyrosetta.pose_from_pdb(pdb_file_path)
            pose_score = scorefxn(pose)
            start_pose_total_score = pyrosetta.rosetta.protocols.relax.get_per_residue_scores(pose, pyrosetta.rosetta.core.scoring.ScoreType.total_score)
            sequence_length = pose.total_residue()
            avg_energy_pre_red = sum(list(start_pose_total_score)) / sequence_length * 1.0
            energy_dict[index] = avg_energy_pre_red
        except:
            continue
    
    relax_score = []
    for i in range(length):
        if i in energy_dict:
            relax_score.append(energy_dict[i])
        else:
            relax_score.append(-1000)
    return relax_score



def load_data(file_path):
       
    labels = []
    sequences = []
    origin_function_scores = []
    plddt_scores = []
    with open(file_path, "r") as file:
        data = file.readlines()

        for line in data:
            line = line.rstrip()
            origin_data = json.loads(line)

            
            label = origin_data[0]
            sequence = origin_data[1]
            plddt = origin_data[3]
            
            
            labels.append(label)
            sequences.append(sequence)
            plddt_scores.append(plddt)
    return labels, sequences, plddt_scores


def find_txt_files(folder_path):

    txt_files = []

    for root, dirs, files in os.walk(folder_path):

        for dirname in dirs:

            file_path = os.path.join(root, dirname, "result.txt")
            txt_files.append( ( dirname, file_path) )
        
    return txt_files



def get_subfolder_paths(dir_path):

    subfolder_paths = []

    for root, dirs, files in os.walk(dir_path):

        for subdir in dirs:

            sub_path = os.path.join(root, subdir)
            subfolder_paths.append(sub_path)

        break
    return subfolder_paths



def get_pdb():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./evaluate_candidate_sequence")
    parser.add_argument("--output_path", type=str, default="./evaluate_candidate_sequence")
    args = parser.parse_args()
    folder_path = args.dataset_path
    
    output_folder_path = args.output_path
    



    
    task_name_list = ["function_0","function_1","process_0","process_1","component_0","component_1"]


    
    
    for task_name in task_name_list:
        input_file_path = os.path.join(folder_path, task_name, "result.txt")
        output_file_path = os.path.join(output_folder_path, task_name, "result.txt")
        output_pdb_folder = os.path.join(output_folder_path, task_name, 'pdb')
        if not os.path.exists(output_pdb_folder):
            os.makedirs(output_pdb_folder)
            
            
            
        labels, sequences, plddt_scores= load_data(input_file_path)
    
        

        
        
        function0_sequences = []
        function0_scores = []
        process0_sequences = []
        process0_scores = []
        component0_sequences = []
        component0_scores = []
        function1_sequences = []
        function1_scores = []
        process1_sequences = []
        process1_scores = []
        component1_sequences = []
        component1_scores = []

        new_function_scores = []
        


        input_structure_path = os.path.join(folder_path, task_name, "structure_embedding")
        training_structure_path = os.path.join("dataset", "structure", task_name, "structure_embedding")
        scores_dict = evaluate_structure_score(input_structure_path, training_structure_path)
        
        new_function_scores = []
        for i in range(len(sequences)):
            if i in scores_dict:
                new_function_scores.append(scores_dict[i])
            else:
                new_function_scores.append(0)
        
        
        relax_file_path = os.path.join(folder_path, task_name,'relax_pdb')
        rosetta_energy_dict = get_rosetta_energy(relax_file_path, len(labels))
        


        
        for i in range(len(sequences)):  
            sequence = sequences[i]
            sequence = sequence.replace('\n','')
            label = labels[i]
            
            if label[0]==0:
                function0_sequences.append(sequence)
            elif label[0] == 1:
                function1_sequences.append(sequence)
            
            if label[1]==0:
                process0_sequences.append(sequence)
            elif label[1] == 1:
                process1_sequences.append(sequence)
               
            if label[2]==0:
                component0_sequences.append(sequence)
            elif label[2] == 1:
                component1_sequences.append(sequence)
                
        func_score = [0 for i in range(len(sequences))]
        if function0_sequences:

            func_score = new_function_scores
        if function1_sequences:

            func_score = new_function_scores
        proc_score = [0 for i in range(len(sequences))]
        if process0_sequences:

            proc_score = new_function_scores
        if process1_sequences:

            proc_score = new_function_scores
        comp_score = [0 for i in range(len(sequences))]
        if component0_sequences:

            comp_score = new_function_scores
        if component1_sequences:

            comp_score = new_function_scores
        
        
        function_score = sum(function0_scores+function1_scores)/len(function0_scores+function1_scores) if len(function0_scores+function1_scores)!=0 else 0
        process_score = sum(process0_scores + process1_scores)/len(process0_scores + process1_scores) if len(process0_scores + process1_scores)!=0 else 0
        component_score = sum(component0_scores + component1_scores)/len(component0_scores + component1_scores) if len(component0_scores + component1_scores)!=0 else 0
        
        
        new_function_score = []
        for i in range(len(sequences)):
            score = [float(func_score[i]), float(proc_score[i]), float(comp_score[i])]
            new_function_score.append(score)
        
        
        
        
        with open(output_file_path,"w") as file:
            for i in range(len(sequences)):
                label = labels[i]
                sequence = sequences[i]
                new_score = new_function_score[i]
                plddt = float(plddt_scores[i]) if plddt_scores[i] else 0

                rosetta_energy = rosetta_energy_dict[i]
                file.write(json.dumps([labels[i], sequences[i].rstrip(), new_score, plddt, rosetta_energy])+'\n')


        
def load_train_data(path):
    sequences = []
    with open(path, 'r') as f:
        next(f)
        for line in f.readlines():
            line = line.rstrip().split("\t")
            sequences.append(line[1].strip())
    return sequences

def get_sentence_embedding(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(sentence_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

def get_dataset_embedding(tokenizer, model, dataset):

    train_encodings = [tokenizer(sequence, max_length=400, padding='max_length', truncation=True, return_tensors="pt") for sequence in dataset]
    train_input_ids = torch.cat([encoding['input_ids'] for encoding in train_encodings], dim=0)
    train_attention_mask = torch.cat([encoding['attention_mask'] for encoding in train_encodings], dim=0)
    batch_size = 512

    train_dataset = TensorDataset(train_input_ids, train_attention_mask)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    train_embeddings = get_sentence_embedding(model, train_dataloader)
    return train_embeddings


def calculate_cosine_similarity(matrix1, matrix2):

    return cosine_similarity(matrix1, matrix2)



if __name__ == "__main__":

    get_pdb()
