
# CtrlProt
Controllable Protein Sequence Generation with LLM Preference Optimization

## Dataset 
The dataset is constructed by extracting specific labels from [UniProtKB](https://www.uniprot.org/) based on Gene Ontology (GO) annotations. The corresponding structures should be downloaded from the [AFDB](https://alphafold.ebi.ac.uk/) and stored in the designated directory.

```python
dataset/
│── component/ # Cellular component data
│ ├── 0.tsv # Cytoplasm (GO:0005737)
│ ├── 1.tsv # Nucleus (GO:0005634)
│
│── function/ # Molecular function data
│ ├── 0.tsv # Metal ion binding (GO:0046872)
│ ├── 1.tsv # RNA binding (GO:0003723)
│
│── process/ # Biological process data
│ ├── 0.tsv # Phosphorylation (GO:0016310)
│ ├── 1.tsv # Translation (GO:0006412)
│
│── structure/ # Protein structures
| ├── component_0/
| | ├── pdb/
| | | ├── *.pdb # PDB files for corresponding proteins
| |
| ├── component_1/
| | ├── pdb/
| | | ├── *.pdb 
| ...
```


## Prerequisites

Before running the scripts, ensure that you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

Other required models and dependencies can be obtained from the following sources:

[ProtGPT2](https://huggingface.co/nferruz/ProtGPT2/tree/main), 
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN), 
[ESMFold](https://huggingface.co/facebook/esmfold_v1),
[ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D),
[PyRosetta](https://www.pyrosetta.org/downloads#h.6vttn15ac69d) and
[Foldseek](https://github.com/steineggerlab/foldseek)

The evaluation dataset and trained classifiers can be downloaded [here](https://huggingface.co/miraitowal/CtrlProt_evaluation_classifiers)

## Run
Here, we show how to run CtrlProt to generate protein sequences with desired attributes.
### 1. Prefix Tuning

Finetune protein language models.

```bash
python prefix_tuning_prot.py --batch_size 16 --epochs 50 --dataset_path ./dataset/function/0.tsv --dataset_name function_0 --output_path ./candidate_prefix_tuning_model/
```

### 2. Candidate Sequences Generation

This script generates candidate sequences using the chosen prefix-tuning model.

```bash
python generate_candidate_sequence.py --model_path ./prefix_tuning_model/
```

### 3. Candidate Sequences Evaluation
We first use ESMFold to predict the structure and get the pdb files of generated proteins.
```bash
python evaluate_candidate_sequence.py --dataset_path ./generate_candidate_sequence --output_path ./evaluate_candidate_sequence
```
Then we use Rosetta Relaxation on Generated Sequences for evaluating structural stability.
```bash
python generate_rosetta_relax.py
```
We use the structural embedding from ProteinMPNN to evaluate the Functionality
```bash
python structure_similarity.py
```
Finally, we use the above score to get the quality score
```bash
python evaluate_candidate_sequence2.py --dataset_path ./evaluate_candidate_sequence --output_path ./mlpo_candidate_sequence
```

### 4. Construct preference optimization dataset and train the model:

These scripts build the dataset.
```bash
python build_mlpo_dataset.py --dataset_path ./mlpo_candidate_sequence

```
We then train the model on the constructed preference optimization dataset.
```bash
python train_mlpo.py --batch_size 16 --epochs 50 --lr 5e-5 --dataset_path ./mlpo_candidate_sequence/function_0/mlpo_dataset --dataset_name function_0 --model_path ./prefix_tuning_model/function_0/
```

### 5.Generation and evaluation

Then we can generate and test the result.
```bash
python single_function_generation.py
```
eval_classifier provides CLS-score, TM-score, rmsd and pLDDT.
```bash
python eval_classifier.py
```

## Citation
```
@inproceedings{CtrlProt,
  title={Controllable Protein Sequence Generation with LLM Preference Optimization},
  author={Liu, Xiangyu and Liu, Yi and Chen, Silei and Hu, Wei},
  booktitle={AAAI},
  year={2025}
}
```






















