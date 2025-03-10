"""
We use pyrosetta for rosetta operations on protein structures, version pyrosetta=2022.41+release.28dc2a1, 
see https://www.pyrosetta.org/downloads for installation details
"""


import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyrosetta import *
import time
import os
dir_path = os.getcwd()
from tqdm import tqdm

pyrosetta.init()
scorefxn =  get_fa_scorefxn()   


relax = pyrosetta.rosetta.protocols.relax.FastRelax()

relax.set_scorefxn(pyrosetta.get_fa_scorefxn())
relax.max_iter(20)

folder = "evaluate_candidate_sequence"

data_dir_name = ["function_0","function_1","process_0", "process_1","component_0","component_1"]


def find_pdb_files(folder_path):

    txt_files = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:
            if '.pdb' in file:
                if os.path.exists(os.path.join(root.replace("pdb","relax_pdb"), file)):
                    continue
                file_path = os.path.join(root, file)
                txt_files.append( ( file, file_path) )
        continue
    return txt_files


start_time = time.time()


def process_pdb(pdb_info):
    pdb_name, pdb_path = pdb_info
    try:
        pdb_id = pdb_name.split('.')[0]
        

        pose = pyrosetta.pose_from_pdb(pdb_path)
        ori_score = scorefxn(pose)
        
        if not os.getenv("DEBUG"):
            relax.apply(pose)
            
        relax_score = scorefxn(pose)
        relax_path = pdb_path.replace("/pdb/", "/relax_pdb/")
        
        if not os.path.exists(os.path.dirname(relax_path)):
            os.makedirs(os.path.dirname(relax_path))
            
        pose.dump_pdb(relax_path)
        
        return pdb_name, ori_score, relax_score, None
    except Exception as e:
        return pdb_name, None, None, str(e)

with ProcessPoolExecutor() as executor:
    futures = []
    for dir_name in data_dir_name:
        pdb_folder_path = os.path.join(folder, dir_name, 'pdb')
        file_list = find_pdb_files(pdb_folder_path)
        

        futures += [executor.submit(process_pdb, pdb_info) for pdb_info in file_list]


    for future in tqdm(as_completed(futures), total=len(futures)):
        pdb_name, ori_score, relax_score, error = future.result()
        if error:
            print(f"Error processing {pdb_name}: {error}")
        else:
            print(f"{pdb_name} original score: {ori_score}, relaxed score: {relax_score}")
    

end_time = time.time()
total_time = end_time - start_time
print("times: ", total_time)


