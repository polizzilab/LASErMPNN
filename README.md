[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/polizzilab/LASErMPNN/blob/main/run_lasermpnn.ipynb)

# LASErMPNN: Small-Molecule Conditioned Protein Sequence Design

### Check out the paper in <i>Nature</i> [here](https://www.nature.com/articles/s41586-026-10670-w)!


![A block diagram of the LASErMPNN architecture depicting information flow through the network.](./images/laser_block_diagram.png)


> [!WARNING]
> A major difference between LASErMPNN and LigandMPNN is that LASErMPNN was trained on protonated structures. 
> Please make sure your ligand has the appropriate hydrogens (in the expected protonation state) attached when running the model or you may encounter unexpected behavior.
> For an example of how to adjust protonation from a smiles string, check out the script `./protonate_and_add_conect_records.py` from the [NISE repo](https://github.com/polizzilab/NISE/blob/main/protonate_and_add_conect_records.py)


### Environment Setup

A minimal version of LASErMPNN could be run in inference mode in any Python environment with PyTorch, torch-scatter, and torch-cluster. 
ProDy is used internally to read and write PDB files.

To ensure your conda installation is using the libmamba solver, run `conda config --show-sources` 
and ensure the output has `solver: libmamba` at the bottom. 
If not, run `conda config --set-solver libmamba`.


The commands below will create an environment called `lasermpnn` which you can then `conda activate lasermpnn` to run the LASErMPNN CLI scripts.

#### CUDA 11 Instructions

To install the training environment on a system running CUDA 11 (check your cuda version by running `nvcc --version`), run the following set of commands using a [MiniForge](https://github.com/conda-forge/miniforge/releases/tag/24.11.3-2) installation (recommended) or an existing conda installation with a libmamba solver (see above).

```bash
conda env create -f conda_env.yml -y
```

#### CUDA 12 Instructions

To install the training environment on a system running CUDA 12 (check your cuda version by running `nvcc --version`), run the following set of commands using a [MiniForge](https://github.com/conda-forge/miniforge/releases/tag/24.11.3-2) installation (recommended) or an existing conda installation with a libmamba solver (see above).

This was tested on a system with CUDA 12.4, you might need to update the specfic CUDA version at the top of the `conda_env_12p4.yml` file if you have a different CUDA 12.x version.

```bash
conda env create -f conda_env_12p4.yml -y
```

### Testing your installation

```
From within the LASErMPNN repository, with the LASErMPNN python environment active, run the command `pytest -s` to run model equivariance checks and a test run of model inference capabilities`. This automatically runs the equivalent of the following to test model inference:
```

```bash
# Clone LASErMPNN
git clone https://github.com/polizzilab/LASErMPNN ./LASErMPNN/

# Assuming LASErMPNN python environment is already installed and active (see above)
# Generate 5 LASErMPNN sequences on example file (runs from parent directory of LASErMPNN installation):
python -m LASErMPNN.run_batch_inference ./LASErMPNN/example_pdbs/4jnj-1_prot.pdb ./LASErMPNN/outputs_dir/ 5
```

### Running Inference

```bash
python -m LASErMPNN.run_inference -h
```

This script outputs a single pdb file named `laser_output.pdb` and is useful for testing:


```text
usage: run_inference.py [-h] [--model_weights MODEL_WEIGHTS] [--output_path OUTPUT_PATH] [--temp SEQUENCE_TEMP] [--fs_sequence_temp FS_SEQUENCE_TEMP] [--bb_noise BACKBONE_NOISE] [--device DEVICE] [--fix_beta] [--ignore_statedict_mismatch] [--ebd] [--repack_only] [--ignore_ligand] [--noncanonical_aa_ligand]
                        [--fs_calc_ca_distance FS_CALC_CA_DISTANCE] [--fs_calc_burial_hull_alpha_value FS_CALC_BURIAL_HULL_ALPHA_VALUE] [--fs_no_calc_burial] [--disable_charged_fs]
                        input_pdb_code

Run LASErMPNN inference on a given PDB file.

positional arguments:
  input_pdb_code        Path to the input PDB file.

options:
  -h, --help            show this help message and exit
  --model_weights MODEL_WEIGHTS, -w MODEL_WEIGHTS
                        Path to dictionary of torch.save()ed model state_dict and training parameters. Default: /nfs/polizzi/bfry/programs/LASErMPNN/model_weights/laser_weights_0p1A_nothing_heldout.pt
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to the output PDB file.
  --temp SEQUENCE_TEMP, -t SEQUENCE_TEMP
                        Sequence sample temperature.
  --fs_sequence_temp FS_SEQUENCE_TEMP, -f FS_SEQUENCE_TEMP
                        Residues around the ligand will be sampled at this temperature, otherwise they default to sequence_temp.
  --bb_noise BACKBONE_NOISE, -n BACKBONE_NOISE
                        Inference backbone noise.
  --device DEVICE, -d DEVICE
                        Pytorch style device string. Ex: "cuda:0" or "cpu".
  --fix_beta, -b        Residues with B-Factor of 1.0 have sequence and rotamer fixed, residues with B-Factor of 0.0 are designed.
  --ignore_statedict_mismatch, -s
                        Small state_dict mismatches are ignored. Don't use this unless any missing parameters aren't learned during training.
  --ebd, -e             Uses entropy based decoding order. Decodes all residues and selects the lowest entropy residue as next to decode, then recomputes all remaining residues. Takes longer than normal decoding.
  --repack_only         Only repack residues, do not design new ones.
  --ignore_ligand       Ignore ligands in the input PDB file.
  --noncanonical_aa_ligand
                        Featurize a noncanonical amino acid as a ligand.
  --fs_calc_ca_distance FS_CALC_CA_DISTANCE
                        Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.
  --fs_calc_burial_hull_alpha_value FS_CALC_BURIAL_HULL_ALPHA_VALUE
                        Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).
  --fs_no_calc_burial   Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance
  --disable_charged_fs  Disable sampling D,K,R,E residues in the first shell around the ligand.
```


### Running Batch Inference

```bash
python -m LASErMPNN.run_batch_inference -h
```

This script is useful to generate multiple designs for one or multiple inputs. Creates an output directory with subdirectories for each input file (unless run with a single input file).

```text
usage: run_batch_inference.py [-h] [--designs_per_batch DESIGNS_PER_BATCH] [--inputs_processed_simultaneously INPUTS_PROCESSED_SIMULTANEOUSLY] [--model_weights_path MODEL_WEIGHTS_PATH] [--sequence_temp SEQUENCE_TEMP] [--first_shell_sequence_temp FIRST_SHELL_SEQUENCE_TEMP]
                              [--chi_temp CHI_TEMP] [--chi_min_p CHI_MIN_P] [--seq_min_p SEQ_MIN_P] [--device INFERENCE_DEVICE] [--use_water] [--silent] [--ignore_key_mismatch] [--disabled_residues DISABLED_RESIDUES] [--fix_beta] [--repack_only_input_sequence] [--ignore_ligand] [-c]
                              [--budget_residue_sele_string BUDGET_RESIDUE_SELE_STRING] [--ala_budget ALA_BUDGET] [--gly_budget GLY_BUDGET] [--noncanonical_aa_ligand] [--repack_all] [--output_fasta] [--output_fasta_only] [--fs_calc_ca_distance FS_CALC_CA_DISTANCE]
                              [--fs_calc_burial_hull_alpha_value FS_CALC_BURIAL_HULL_ALPHA_VALUE] [--fs_no_calc_burial] [--disable_charged_fs]
                              input_pdb_directory output_pdb_directory designs_per_input

Run batch LASErMPNN inference.

positional arguments:
  input_pdb_directory   Path to directory of input .pdb or .pdb.gz files, a single input .pdb or .pdb.gz file, or a .txt file of paths to input .pdb or .pdb.gz files.
  output_pdb_directory  Path to directory to output LASErMPNN designs.
  designs_per_input     Number of designs to generate per input.

options:
  -h, --help            show this help message and exit
  --designs_per_batch DESIGNS_PER_BATCH, -b DESIGNS_PER_BATCH
                        Number of designs to generate per batch. If designs_per_input > designs_per_batch, chunks up the inference calls in batches of this size. Default is 30, can increase/decrease depending on available GPU memory.
  --inputs_processed_simultaneously INPUTS_PROCESSED_SIMULTANEOUSLY, -n INPUTS_PROCESSED_SIMULTANEOUSLY
                        When passed a list of multiple files, this is the number of input files to process per pass through the GPU. Useful when generating a few sequences for many input files.
  --model_weights_path MODEL_WEIGHTS_PATH, -w MODEL_WEIGHTS_PATH
                        Path to model weights. Default: /nfs/polizzi/bfry/programs/LASErMPNN/model_weights/laser_weights_0p1A_nothing_heldout.pt. Other weights can be found in the ./model_weights/ directory.
  --sequence_temp SEQUENCE_TEMP
                        Temperature for sequence sampling.
  --first_shell_sequence_temp FIRST_SHELL_SEQUENCE_TEMP
                        Temperature for first shell sequence sampling. Can be used to disentangle binding site temperature from global sequence temperature for harder folds.
  --chi_temp CHI_TEMP   Temperature for chi sampling.
  --chi_min_p CHI_MIN_P
                        Minimum probability for chi sampling. Not recommended.
  --seq_min_p SEQ_MIN_P
                        Minimum probability for sequence sampling. Not recommended.
  --device INFERENCE_DEVICE, -d INFERENCE_DEVICE
                        PyTorch style device string (e.g. "cuda:0").
  --use_water           Parses water (resname HOH) as part of a ligand.
  --silent              Silences all output except pbar.
  --ignore_key_mismatch
                        Allows mismatched keys in checkpoint statedict
  --disabled_residues DISABLED_RESIDUES
                        Residues to disable in sampling.
  --fix_beta            If B-factors are set to 1, fixes the residue and rotamer, if not, designs that position.
  --repack_only_input_sequence
                        Repacks the input sequence without changing the sequence.
  --ignore_ligand       Ignore ligand in sampling.
  -c, --constrain_ala_gly_sampling_to_exposed_non_secondary_structure
                        If set, constrains number of ALA and GLY residues that can be sampled in exposed non-secondary structured residues. The max number of ALA and GLY residues is set by the --ala_budget and --gly_budget arguments.
  --budget_residue_sele_string BUDGET_RESIDUE_SELE_STRING
                        A ProDy-style selection string to constrain the residues to sample ALA and GLY residues in. Can be used to override the automatic selection performed by the --constrain_ala_gly_sampling_to_exposed_non_secondary_structure flag.
  --ala_budget ALA_BUDGET
                        Maximum number of ALA residues that can be sampled in exposed non-secondary structured residues. Only used if --constrain_ala_gly_sampling_to_exposed_non_secondary_structure is set or --budget_residue_sele_string is set. Default is 4.
  --gly_budget GLY_BUDGET
                        Maximum number of GLY residues that can be sampled in exposed non-secondary structured residues. Only used if --constrain_ala_gly_sampling_to_exposed_non_secondary_structure is set or --budget_residue_sele_string is set. Default is 0.
  --noncanonical_aa_ligand
                        Featurize a noncanonical amino acid as a ligand.
  --repack_all          Repack all residues, even those with chain_mask=1.
  --output_fasta        Output a fasta file of the designed sequences in addition to the PDB files.
  --output_fasta_only   Output only a fasta file of the designed sequences, does not write PDB files.
  --fs_calc_ca_distance FS_CALC_CA_DISTANCE
                        Distance between a ligand heavy atom and CA carbon to consider that carbon first shell.
  --fs_calc_burial_hull_alpha_value FS_CALC_BURIAL_HULL_ALPHA_VALUE
                        Alpha parameter for defining convex hull. May want to try setting to larger values if using folds with larger cavities (ex: ~100.0).
  --fs_no_calc_burial   Disable using a burial calculation when selecting first shell residues, if true uses only distance from --fs_calc_ca_distance
  --disable_charged_fs  Disable sampling D,K,R,E residues in the first shell around the ligand.
```


### Training LASErMPNN

To retrain the model, download the datasets with `download_ligand_encoder_training_dataset.sh` and `download_protonated_pdb_training_dataset.sh` for each respective dataset by running them in the project's root directory.

We used 4x A6000 GPUs to train the LASErMPNN model which takes around 24 hrs for 60k optimizer steps. See `train_lasermpnn.py` for more information.


### Training Ligand Encoder

Training the Ligand Encoder module can be done with much lower memory and a single GPU. See `pretrain_ligand_encoder.py` for more information.


### Neural Iterative Selection & Expansion Implementation

see https://www.github.com/polizzilab/NISE for a NISE protocol implementation using Boltz-1x/2x.


### Re-training LigandMPNN

The code for retraining the LigandMPNN architecture on the streptavidin heldout split and reconstruction of the ligandmpnn training dataset are available as files suffixed with `_ligandmpnn`.
We did not reimplement the LigandMPNN Sidechain Packer (only the sequence generation model) so the `.pdb` formatted outputs from sequence design with a retrained lignadmpnn model will have sidechains with all dihedral angles fixed to values of 0.0.
It may be more useful to run any retrained ligandmpnn models using the `--output_fasta_only` flags since the predicted sidechains contain no useful information other than for threading the sequence onto the input backbone.

To retrain the LigandMPNN model in the same way we tested it in the paper, follow the instructions above for Training LASErMPNN and see `train_ligandmpnn.py` for more information.
