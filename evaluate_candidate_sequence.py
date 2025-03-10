import os
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
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
def calculate_average_plddt_from_pdb(pdb_file_path):

    structure = parser.get_structure('protein', pdb_file_path)
    
    plddt_values = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    plddt_values.append(atom.bfactor)
    
    if not plddt_values:
        return None
    
    average_plddt = sum(plddt_values) / len(plddt_values)
    return average_plddt



torch.backends.cuda.matmul.allow_tf32 = True



class Structure_Evaluate():
    def __init__(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained("/data/anonymity/Pre_Train_Model/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("/data/anonymity/Pre_Train_Model/esmfold_v1", low_cpu_mem_usage=True)
        
        self.model = self.model.cuda()
        
    def get_plddt_scores(self, output_pdb_folder, sequences,num = -1):
        has_finished = [file for file in os.listdir(output_pdb_folder)]
        plddt_results = []
        pdb_results = []

        if num == -1:
            temp_sequences = sequences
        else:
            temp_sequences = sequences[ : num]
        for i in tqdm(range(len(temp_sequences))):
            if f"{i}.pdb" in has_finished:
                pdb = None
                temp_path = os.path.join(output_pdb_folder, f"{i}.pdb")
                if os.path.getsize(temp_path) != 0:
                    plddt_score = calculate_average_plddt_from_pdb(os.path.join(output_pdb_folder, f"{i}.pdb"))
                    pdb_results.append(pdb)
                    plddt_results.append(plddt_score)
                    continue
            test_protein = temp_sequences[i]

            try:
                tokenized_input = self.tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)['input_ids']
                tokenized_input = tokenized_input.cuda()
                with torch.no_grad():
                    outputs = self.model(tokenized_input)
                    pdb = self.convert_outputs_to_pdb(outputs)
                    plddt_score = self.compute_average_plddt(outputs)
                plddt_results.append(plddt_score)
                pdb_results.append(pdb[0])
                
                output_pdb_file = os.path.join(output_pdb_folder, f"{i}.pdb")
                with open(output_pdb_file, 'w') as file:
                    file.write(pdb[0])
                print(f"create pdb file {i}.pdb")
            except:
                with open("/data1/anonymity/CtrlProt/wrong_sample.txt",'a') as file:
                    file.write(test_protein + '\n')
                plddt_results.append(0)
                pdb_results.append(None)

            
        return plddt_results, pdb_results
    
    
    def compute_average_plddt(self, outputs):
        x = outputs['plddt']
        return torch.mean(outputs['plddt']).item()
        
    def convert_outputs_to_pdb(self, outputs):
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"]
        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i],
                chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))
        return pdbs
    
def load_data(file_path):
       
    labels = []
    sequences = []
    with open(file_path, "r") as file:
        data = file.readlines()

        for line in data:
            line = line.rstrip()
            origin_data = json.loads(line)

            
            label = origin_data[0]
            sequence = origin_data[1]
            
            labels.append(label)
            sequences.append(sequence)
    return labels, sequences


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
    parser.add_argument("--dataset_path", type=str, default="./generate_candidate_sequence")
    parser.add_argument("--output_path", type=str, default="./evaluate_candidate_sequence")
    args = parser.parse_args()
    folder_path = args.dataset_path
    
    output_folder_path = args.output_path

    device = "cuda" if cuda.is_available() else "cpu"

    
    structure_evaluate = Structure_Evaluate()
    
    task_name_list = ["function_0","function_1","process_0","process_1","component_0","component_1"]
    for task_name in task_name_list:
        input_file_path = os.path.join(folder_path, task_name, "result.txt")
        output_file_path = os.path.join(output_folder_path, task_name, "result.txt")
        output_pdb_folder = os.path.join(output_folder_path, task_name, 'pdb')
        if not os.path.exists(output_pdb_folder):
            os.makedirs(output_pdb_folder)
            
            
            
        labels, sequences = load_data(input_file_path)
        
        plddt_scores = []
        
        
        plddt_scores, pdb_results = structure_evaluate.get_plddt_scores(output_pdb_folder, sequences)
        


        with open(output_file_path,"w") as file:
            for i in range(len(sequences)):
                label = labels[i]
                sequence = sequences[i]
                plddt = float(plddt_scores[i]) if plddt_scores[i] else 0
                file.write(json.dumps([labels[i], sequences[i].rstrip(), plddt])+'\n')



if __name__ == "__main__":
    get_pdb()
