import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
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
import py3Dmol
from IPython.display import display
import subprocess
import tempfile
torch.backends.cuda.matmul.allow_tf32 = True


class Classifier:
    def __init__(self, model_checkpoint, device):
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint).to(self.device)
    
    def preprocess_data(self, sequences):
        batch_encoding = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset.from_dict(batch_encoding)
        return dataset

    def classify(self, sequences):
        dataset = self.preprocess_data(sequences)
        dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=default_data_collator, batch_size=8, pin_memory=True)
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device) 
                attention_mask = batch["attention_mask"].to(self.device) 
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                scores = probabilities[:, 1]
                results.extend(scores.cpu().numpy())
        return results


class Structure_Evaluate():
    def __init__(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained("/data1/anonymity/Pre_Train_Model/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("/data1/anonymity/Pre_Train_Model/esmfold_v1", low_cpu_mem_usage=True)

        self.model = self.model.cuda()
        
    def get_plddt_scores(self, task_name1, task_name2, sequences,num = -1):
        results = []
        tmscore1_list = []
        rmsd1_list = []
        
        tmscore2_list = []
        rmsd2_list = []
        
        temp_sequences = sequences[ : num]
        
        db_path1 = os.path.join("/data1/anonymity/foldseek", task_name1)
        db_path2 = os.path.join("/data1/anonymity/foldseek", task_name2)
        for test_protein in temp_sequences:
            

            try:
                tokenized_input = self.tokenizer([test_protein], return_tensors="pt", add_special_tokens=False)['input_ids']
                tokenized_input = tokenized_input.cuda()
                with torch.no_grad():
                    outputs = self.model(tokenized_input)
                    pdb_content = self.convert_outputs_to_pdb(outputs)
                    pdb_content = pdb_content[0]
                    tmscore1, rmsd1 = self.get_foldseek_result(pdb_content, db_path1)
                    tmscore2, rmsd2 = self.get_foldseek_result(pdb_content, db_path2)
                tmscore1_list.append(tmscore1)
                rmsd1_list.append(rmsd1)
                tmscore2_list.append(tmscore2)
                rmsd2_list.append(rmsd2)
                results.append(self.compute_average_plddt(outputs))
            except:
                x=1
            
        
        return results, tmscore1_list, rmsd1_list, tmscore1_list, rmsd1_list
    
    
    def compute_average_plddt(self, outputs):
        x = outputs['plddt']
        return torch.mean(outputs['plddt']).item()
        
    def get_foldseek_result(self,pdb_content, db_path):

        output_content = self.run_foldseek(pdb_content, db_path)
        output_content = output_content.split("\n")[0]
        result = output_content.split("\t")
        name_list = ["query","target","fident","prob","lddt","alntmscore","qtmscore","ttmscore","rmsd"]
        
        result = dict(zip(name_list, result))
    
        return float(result["qtmscore"]), float(result["rmsd"])
    
        
    def run_foldseek(self, pdb_content, db_path):

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdb') as tmp_query:
            tmp_query.write(pdb_content)
            tmp_query_path = tmp_query.name
        

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.aln') as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            
            command = [
                "foldseek", "easy-search",
                tmp_query_path, db_path, tmp_output_path, "../tmp",
                "--format-output", "query,target,fident,prob,lddt,alntmscore,qtmscore,ttmscore,rmsd"
            ]
            with open(os.devnull, 'w') as devnull:
                
                subprocess.run(command, check=True, stdout=devnull, stderr=devnull)

            with open(tmp_output_path, 'r') as f:
                output_content = f.read()
        finally:

            os.remove(tmp_query_path)
            os.remove(tmp_output_path)   
        return output_content  

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



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./single_function_result")
    args = parser.parse_args()
    folder_path = args.dataset_path


    task_name = ["process_0", "process_1"]
    task_name = ["component_0", "component_1"]
    task_name = ["function_0", "function_1"]
    task_name = ["function_1"]
    

    task_name = [("function_0","component_1")]

    
    
    
    result_path = os.path.join(folder_path, 'score.txt')
    with open(result_path,"w") as file:

        for (task1, task2)  in task_name:
            folder_path = os.path.join(args.dataset_path,  task1+"_"+task2)
            
            subfolder_list = get_subfolder_paths(folder_path)
            subfolder_list = sorted(subfolder_list, key=lambda x: int(x.split('-')[-1]))
            
            
            process0_model_checkpoint = "/data1/anonymity/Pre_Train_Model/protein_evaluation/process_0"
            function0_model_checkpoint = "/data1/anonymity/Pre_Train_Model/protein_evaluation/function_0"
            component0_model_checkpoint = "/data1/anonymity/Pre_Train_Model/protein_evaluation/component_0"
            
            process1_model_checkpoint = "/data1/anonymity/Pre_Train_Model/protein_evaluation/process_1"
            function1_model_checkpoint = "/data1/anonymity/Pre_Train_Model/protein_evaluation/function_1"
            component1_model_checkpoint = "/data1/anonymity/Pre_Train_Model/protein_evaluation/component_1"
            
            
            device = "cuda" if cuda.is_available() else "cpu"
            process0_classifier = Classifier(process0_model_checkpoint, device)
            function0_classifier = Classifier(function0_model_checkpoint, device)
            component0_classifier = Classifier(component0_model_checkpoint, device)
            
            process1_classifier = Classifier(process1_model_checkpoint, device)
            function1_classifier = Classifier(function1_model_checkpoint, device)
            component1_classifier = Classifier(component1_model_checkpoint, device)
            
            structure_evaluate = Structure_Evaluate()
            
            result_path = os.path.join(folder_path, 'score.txt')
            with open(result_path,"w") as file:
                for subfolder in tqdm(subfolder_list):
                    

                    file_list = [os.path.join(subfolder,'result.txt')]
                    for file_path in tqdm(file_list):   
                        file_name = os.path.basename(subfolder)
                        

                    

                        labels, sequences = load_data(file_path)
                        
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
                        plddt_scores = []
                        tmscores1 = []
                        rmsds1 = []
                        tmscores2 = []
                        rmsds2 = []
                        
                        

                        plddt_scores, tmscores1, rmsds1, tmscores2, rmsds2 = structure_evaluate.get_plddt_scores(task1, task2, sequences, num = 10)
                        
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
                                
                        
                        if function0_sequences:
                            function0_scores = function0_classifier.classify(function0_sequences)       
                        if process0_sequences:
                            process0_scores = process0_classifier.classify(process0_sequences)      
                        if component0_sequences:
                            component0_scores = component0_classifier.classify(component0_sequences)
                        
                        if function1_sequences:
                            function1_scores = function1_classifier.classify(function1_sequences)       
                        if process1_sequences:
                            process1_scores = process1_classifier.classify(process1_sequences)      
                        if component1_sequences:
                            component1_scores = component1_classifier.classify(component1_sequences)
                        
                        
                        
                        function_score = sum(function0_scores+function1_scores)/len(function0_scores+function1_scores) if len(function0_scores+function1_scores)!=0 else 0
                        process_score = sum(process0_scores + process1_scores)/len(process0_scores + process1_scores) if len(process0_scores + process1_scores)!=0 else 0
                        component_score = sum(component0_scores + component1_scores)/len(component0_scores + component1_scores) if len(component0_scores + component1_scores)!=0 else 0
                        
                        
                        avg_plddt_score = sum(plddt_scores)/len(plddt_scores) if len(plddt_scores)!=0 else 0
                        avg_tmscore1 = sum(tmscores1)/len(tmscores1) if len(tmscores1)!=0 else 0
                        avg_rmsd1 = sum(rmsds1)/len(rmsds1) if len(rmsds1)!=0 else 0
                        
                        avg_tmscore2 = sum(tmscores2)/len(tmscores2) if len(tmscores2)!=0 else 0
                        avg_rmsd2 = sum(rmsds2)/len(rmsds2) if len(rmsds2)!=0 else 0
                        

                        file.write(f"{file_name}, function:{function_score}, process:{process_score}, component:{component_score}, plddt:{avg_plddt_score}, tmscore1:{avg_tmscore1}, rmsd1:{avg_rmsd1}, tmscore2:{avg_tmscore2}, rmsd2:{avg_rmsd2}\n")
                
                        
                
            


if __name__ == "__main__":
    main()
