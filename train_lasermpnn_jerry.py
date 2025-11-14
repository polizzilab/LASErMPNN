#!/usr/bin/env python
"""
Train LASErMPNN model with multiple GPUs using PyTorch distributed training.
Set the visible device indices immediately.

"""
import os
from pathlib import Path

if __name__ == '__main__':
    # Need to do this before importing any torch modules.
    # VISIBLE_DEVICES = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    VISIBLE_DEVICES = ['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([x.split(':')[-1] for x in VISIBLE_DEVICES])

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict

from LASErMPNN.utils.model import LASErMPNN
from LASErMPNN.utils.optimizer import get_std_opt, NoamOpt
from LASErMPNN.utils.helper_functions import compute_sidechain_rmsd
from LASErMPNN.utils.build_rotamers import compute_chi_angle_accuracies
from LASErMPNN.utils.constants import aa_short_to_idx, aa_idx_to_short, aa_to_chi_angle_atom_map
from LASErMPNN.utils.pdb_dataset import UnclusteredProteinChainDataset, JerryDatasetSampler, collate_sampler_data, BatchData, invert_dict, chain_list_to_protein_chain_dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean, scatter

# Distributed training imports.
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, Dataset, Sampler
from torch.nn.parallel import DistributedDataParallel
from operator import itemgetter
from torch import autocast # type: ignore
import traceback

from typing import *

CURR_FILE_DIR_PATH = Path(__file__).parent


class TrainingFinishedException(Exception):
    pass


class DatasetFromSampler(Dataset):
    """
    Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler

    https://github.com/singlaayush/MINIT/blob/main/distributed_sampler_wrapper.py
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler) # type: ignore


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.

    https://github.com/singlaayush/MINIT/blob/main/distributed_sampler_wrapper.py
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int],
        rank: Optional[int],
        shuffle: bool,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        if len(self.dataset) == 1:
            return iter(list(self.sampler))
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes)) # type: ignore


def fix_ligand_encoder_weights(model: LASErMPNN, ligand_encoder_state_dict: dict, fix_ligand_encoder_weights: bool) -> None:
    if not fix_ligand_encoder_weights:
        return

    # Load pretrained ligand encoder weights.
    ligand_enc_weights = {x:y for x,y in ligand_encoder_state_dict.items() if x.startswith('ligand_encoder.')}
    model.load_state_dict(ligand_enc_weights, strict=False)
    for param in model.ligand_encoder.parameters():
        param.requires_grad = False


def setup_distributed(rank: int, world_size: int, master_port: str):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # Initialize distributed training.
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)


def prepare_dataloaders(rank, world_size, params, epoch_num, dataset, collate_fn, test_only=False):
    """
    Initializes a distributed dataloader 
    """
    seed = epoch_num
    if params['random_seed'] is not None:
        seed += params['random_seed']

    # The sampler is responsible for shuffling the dataset according to the current epoch in a way that should be deterministic between workers.
    # The DistributedSamplerWrapper divides current samples between workers.
    test_sampler = JerryDatasetSampler(dataset, params, is_train=False, seed=seed, max_protein_length=params['max_protein_size'])
    test_dist_sampler = DistributedSamplerWrapper(test_sampler, num_replicas=world_size, rank=rank, shuffle=False)
    test_dataloader = DataLoader(dataset, batch_sampler=test_dist_sampler, collate_fn=collate_fn)

    if test_only:
        return test_dataloader

    train_sampler = JerryDatasetSampler(dataset, params, is_train=True, seed=seed, max_protein_length=params['max_protein_size'])
    train_dist_sampler = DistributedSamplerWrapper(train_sampler, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(dataset, batch_sampler=train_dist_sampler, collate_fn=collate_fn) 

    # Sanity check that sampler returns same indices for all workers.
    # print(f'GPU-{rank}', list(train_sampler))

    return train_dataloader, test_dataloader


def compute_sequence_loss(sequence_logits: torch.Tensor, sequence_indices: torch.Tensor, valid_residue_mask: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy loss for sequence prediction.
    """
    loss = F.cross_entropy(sequence_logits[valid_residue_mask], sequence_indices[valid_residue_mask], reduction='sum')
    # return loss / max(valid_residue_mask.sum().cpu().item(), 1)
    return loss


def compute_msa_loss(sequence_logits: torch.Tensor, msa_matrix: torch.Tensor, valid_residue_mask: torch.Tensor, correct_sequence_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalizes predicting conserved residues in MSA less than non-conserved residues only for incorrectly predicted residues..
    May help model to make better incorrect predictions.
    """
    incorrect_sequence_mask = ~correct_sequence_mask
    loss = F.cross_entropy(sequence_logits[valid_residue_mask][incorrect_sequence_mask], msa_matrix[valid_residue_mask][incorrect_sequence_mask], reduction='sum') 
    # return loss / max((incorrect_sequence_mask).sum().cpu().item(), 1)
    return loss


def compute_chi_loss(chi_logits: torch.Tensor, chi_angles: torch.Tensor, valid_residue_mask: torch.Tensor, chi_mask: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy loss for chi-angle prediction.
    """
    # Remove invalid residues from chi_mask.
    chi_mask = chi_mask[valid_residue_mask]
    if chi_mask.shape[0] == 0:
        return torch.tensor(0.0, device=chi_logits.device)

    # Compute cross entropy loss.
    loss = F.cross_entropy(chi_logits[valid_residue_mask][chi_mask], chi_angles[valid_residue_mask][chi_mask].argmax(dim=-1), reduction='sum')
    # return loss / max(chi_mask.any(dim=-1).sum().item(), 1)
    return loss

def compute_chi_offset_loss(offset_logits, target_offsets, valid_residue_mask):
    chi_mask = ~(target_offsets[valid_residue_mask].isnan())
    loss = F.mse_loss(offset_logits[valid_residue_mask][chi_mask], target_offsets[valid_residue_mask][chi_mask], reduction='sum')
    return loss


@torch.no_grad()
def sample_dataset(rank, model, dataloader: DataLoader, epoch_num: int, params: dict) -> dict:
    """
    Implements autoregressive sampling of dataloader dataset.
    Fully designs protein sequence and rotamers for each residue according to a randomized decoding order.
    Evaluates batched residues in parallel.
    """
    model.eval()

    # Initialize epoch data.
    output_ints = defaultdict(int)
    output_floats = defaultdict(float)
    rmsd_output_lists = defaultdict(list)
    chi_output_lists = defaultdict(list)
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Sampling test dataset for test epoch {epoch_num}", leave=False, dynamic_ncols=True, disable=(rank != 0)): # type: ignore
        batch: BatchData

        # Skip some batches to speed up sampling.
        if np.random.rand() < (1 - params['autoregressive_test_batch_sample_fraction']):
            continue

        # Prepare batch for model input.
        batch.to_device(model.module.device)
        batch.sample_pseudoligands(100_000, 100_000, model.module.rotamer_builder)
        if params['recompute_all_cb_atoms']:
            batch.recompute_cb_atoms()
        batch.construct_graphs(
            model.module.rotamer_builder, 
            model.module.ligand_featurizer, 
            **params['model_params']['graph_structure'], 
            protein_training_noise=0.0, 
            ligand_training_noise=0.0,
            subgraph_only_dropout_rate=0.0,
            num_adjacent_residues_to_drop=params['sampled_residue_adjacent_drop_num'],
            build_hydrogens=params['model_params']['build_hydrogens'],
            use_aliphatic_ligand_hydrogens=params['use_aliphatic_ligand_hydrogens'],
        )
        batch.generate_decoding_order(stack_tensors = True)

        valid_residue_mask = (batch.sampled_chain_mask) & (~batch.extra_atom_contact_mask) & (~(batch.sequence_indices == aa_short_to_idx['X']))
        if int(valid_residue_mask.sum().item()) == 0:
            continue

        # Run model sample function pass, prevent sampling of X residues.
        sampled_output = model.module.sample(batch, disabled_residues=['X'], disable_pbar=True)

        # Get the subset of residues that were sampled correctly.
        correct_sequence_mask = (batch.sequence_indices == sampled_output.sampled_sequence_indices) & valid_residue_mask
        valid_first_shell_mask = batch.first_shell_ligand_contact_mask & valid_residue_mask
        correct_sequence_first_shell_mask = correct_sequence_mask & valid_first_shell_mask

        # Compute RMSD of sampled sidechain atoms to ground-truth atoms.
        eval_backbones = batch.backbone_coords[correct_sequence_mask]
        eval_sequence = batch.sequence_indices[correct_sequence_mask]
        eval_label_chi = batch.chi_angles[correct_sequence_mask]
        eval_pred_chi = sampled_output.sampled_chi_degrees[correct_sequence_mask]
        ground_truth_coords, backbone_alignment_matrices = model.module.rotamer_builder.build_rotamers(eval_backbones, eval_label_chi, eval_sequence, return_backbone_alignment_matrices=True) # type: ignore
        generated_coords = model.module.rotamer_builder.build_rotamers(eval_backbones, eval_pred_chi, eval_sequence, backbone_alignment_matrices=backbone_alignment_matrices) # type: ignore
        rmsd_metadata_dict = compute_sidechain_rmsd(ground_truth_coords, generated_coords, eval_sequence) # type: ignore
        for aa_code, rmsd in rmsd_metadata_dict.items():
            rmsd_output_lists[f'{aa_code}_rmsd'].append(rmsd)

        # Compute cumulative chi-angle accuracy.
        chi_accuracy = compute_chi_angle_accuracies(sampled_output.sampled_chi_degrees[correct_sequence_mask & valid_residue_mask], batch.chi_angles[correct_sequence_mask & valid_residue_mask], model.module.rotamer_builder)
        for chi_acc, acc_value in chi_accuracy.items():
            chi_output_lists[chi_acc].append(acc_value)

        output_ints['num_sequence_correct_first_shell'] += int(correct_sequence_first_shell_mask.sum().item())
        output_ints['num_valid_first_shell_residues'] += int(valid_first_shell_mask.sum().item())
        output_ints['num_sequence_correct'] += int(correct_sequence_mask[valid_residue_mask].sum().item())
        output_ints['num_valid_residues'] += int(valid_residue_mask.sum().item())
        output_ints['num_batches'] += 1

    # Compute mean of all lists
    rmsd_output_lists = {x: sum(y) / max(len(y), 1) for x,y in rmsd_output_lists.items()}
    chi_output_lists = {x: sum(y) / max(len(y), 1) for x,y in chi_output_lists.items()}
    output_floats['sequence_recovery'] = output_ints['num_sequence_correct'] / max(output_ints['num_valid_residues'], 1)
    output_floats['first_shell_sequence_recovery'] = output_ints['num_sequence_correct_first_shell'] / max(output_ints['num_valid_first_shell_residues'], 1)

    output = dict()
    output.update(dict(output_ints))
    output.update(dict(output_floats))
    output['rmsd'] = rmsd_output_lists
    output['chi_accuracy'] = chi_output_lists
    return {f'sampled_{x}': y for x,y in output.items()}


def save_model_checkpoint(model, params, ligand_enc_params, optimizer, optimizer_steps, epoch_num) -> None:
    """
    Writes the model and optimizer state to disk at prefix provided in params['output_weights_checkpoint_prefix'].
    """
    checkpoint = {
        'params': params,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict(),
        'optimizer_num_steps': optimizer_steps,
        'optimizer_rate': optimizer._rate,
        'ligand_encoder_params': ligand_enc_params,
        'resume_epoch': epoch_num + 1,
    }
    torch.save(checkpoint, f'{params["output_weights_checkpoint_prefix"]}_optstep_{optimizer_steps}.pt')


def average_between_dicts(dicts: List[dict]):
    """
    Averages values between dictionaries.
    """
    output = defaultdict(list)
    all_keys_set = set().union(*dicts)
    for key in all_keys_set:
        for subdict in dicts:
            if key in subdict:
                output[key].append(subdict[key])
    return {x: sum(y) / len(dicts) for x,y in output.items()}


@torch.no_grad()
def test(rank, test_dataloader, model, params):
    model.eval()

    int_log = defaultdict(int)
    list_log = defaultdict(list)
    for batch_idx, batch in enumerate(tqdm(test_dataloader, disable=(rank != 0), desc='Testing', dynamic_ncols=True, leave=False)):
        batch: BatchData

        batch.to_device(rank)
        batch.sample_pseudoligands(100_000, 100_000, model.module.rotamer_builder)
        if params['recompute_all_cb_atoms']:
            batch.recompute_cb_atoms()
        batch.construct_graphs(
            model.module.rotamer_builder,  # type: ignore
            model.module.ligand_featurizer, # type: ignore
            **params['model_params']['graph_structure'], 
            protein_training_noise=0.0, 
            ligand_training_noise=0.0,
            subgraph_only_dropout_rate=0.0,
            num_adjacent_residues_to_drop=params['sampled_residue_adjacent_drop_num'],
            build_hydrogens=params['model_params']['build_hydrogens'],
            use_aliphatic_ligand_hydrogens=params['use_aliphatic_ligand_hydrogens'],
        )
        batch.generate_decoding_order()
        sequence_logits, _, chi_logits, chi_offset_pred, true_chi_offsets = model(batch)
        valid_residue_mask = (batch.sampled_chain_mask) & ((~batch.extra_atom_contact_mask) | (batch.sequence_indices == aa_short_to_idx['C']))

        if valid_residue_mask.sum().item() == 0:
            continue

        # Compute sequence loss.
        seq_loss = compute_sequence_loss(sequence_logits, batch.sequence_indices, valid_residue_mask) / valid_residue_mask.sum()

        # Compute chi loss.
        chi_mask = ~batch.chi_angles.isnan()
        num_valid_chi = (~chi_mask[valid_residue_mask].isnan()).sum()
        chi_encoding = model.module.rotamer_builder.compute_binned_degree_basis_function(batch.chi_angles).nan_to_num() # type: ignore
        chi_loss = compute_chi_loss(chi_logits, chi_encoding, valid_residue_mask, chi_mask) / num_valid_chi

        chi_offset_loss = params['chi_offset_loss_weight'] * compute_chi_offset_loss(chi_offset_pred, true_chi_offsets, valid_residue_mask) / num_valid_chi

        # Compute MSA loss.
        sequence_correct_mask = sequence_logits.argmax(dim=-1)[valid_residue_mask] == batch.sequence_indices[valid_residue_mask]
        msa_loss = params['msa_loss_weight'] * compute_msa_loss(sequence_logits, batch.msa_data, valid_residue_mask, sequence_correct_mask) / valid_residue_mask.sum()

        # Compute loss.
        loss = seq_loss + chi_loss + msa_loss + chi_offset_loss

        # Track accuracy.
        valid_first_shell_mask = batch.first_shell_ligand_contact_mask & valid_residue_mask
        num_correct = (sequence_logits[valid_residue_mask].argmax(dim=-1) == batch.sequence_indices[valid_residue_mask]).sum().item()
        num_residues = int(valid_residue_mask.sum().item())
        num_fs_correct = (sequence_logits[valid_first_shell_mask].argmax(dim=-1) == batch.sequence_indices[valid_first_shell_mask]).sum().item()
        num_fs_residues = int(valid_first_shell_mask.sum().item())

        list_log['loss'].append(loss.item())
        list_log['seq_loss'].append(seq_loss.item())
        list_log['chi_loss'].append(chi_loss.item())
        list_log['msa_loss'].append(msa_loss.item())
        list_log['chi_offset_loss'].append(chi_offset_loss.item())
        int_log['num_correct'] += num_correct
        int_log['num_residues'] += num_residues
        int_log['num_fs_correct'] += num_fs_correct
        int_log['num_fs_residues'] += num_fs_residues

    output = {}
    output['loss'] = sum(list_log['loss']) / max(len(list_log['loss']), 1)
    output['seq_loss'] = sum(list_log['seq_loss']) / max(len(list_log['seq_loss']), 1)
    output['chi_loss'] = sum(list_log['chi_loss']) / max(len(list_log['chi_loss']), 1)
    output['msa_loss'] = sum(list_log['msa_loss']) / max(len(list_log['msa_loss']), 1)
    output['chi_offset_loss'] = sum(list_log['chi_offset_loss']) / max(len(list_log['chi_offset_loss']), 1)
    output['accuracy'] = int_log['num_correct'] / max(int_log['num_residues'], 1)
    output['fs_accuracy'] = int_log['num_fs_correct'] / max(int_log['num_fs_residues'], 1)

    return output

def train(rank, world_size, params):

    starting_epoch_idx = 0
    checkpoint_dict = None
    if params['checkpoint_to_resume_from'] is not None: 
        checkpoint_dict = torch.load(params['checkpoint_to_resume_from'], map_location=f'cuda:{rank}', weights_only=True)
        starting_epoch_idx = checkpoint_dict['resume_epoch'] - 1

    # Starts wandb log if first device and wandb enabled.
    if rank == 0 and params['use_wandb']:
        wandb.init(project='distributed-laser', entity='benf549', config=params)

    # Setup process group.
    setup_distributed(rank, world_size, params['master_port'])

    # Initialize model and optimizer.
    device = torch.device(f'cuda:{rank}')
    if isinstance(params['pretrained_ligand_encoder_weights'], os.PathLike):

        if not os.path.exists(params['pretrained_ligand_encoder_weights']):
            raise FileNotFoundError(f"Pretrained ligand encoder weights not found at {params['pretrained_ligand_encoder_weights']}.")

        ligand_encoder_checkpoint = torch.load(params['pretrained_ligand_encoder_weights'], map_location=device, weights_only=True)
        ligand_enc_params = ligand_encoder_checkpoint['params']
        checkpoint_state_dict = ligand_encoder_checkpoint['model_state_dict']

    elif isinstance(params['pretrained_ligand_encoder_weights'], dict):
        ligand_enc_params = params['pretrained_ligand_encoder_weights']
        checkpoint_state_dict = None
    else:
        raise ValueError(f"params['pretrained_ligand_encoder_weights'] must be a string or a dictionary of params to train a new ligand encoder.")

    model = LASErMPNN(ligand_encoder_params=ligand_enc_params, **params['model_params']).to(rank)
    params['model_params']['graph_structure']['lig_lig_knn_graph_k'] = ligand_enc_params['model_params']['graph_structure']['lig_lig_knn_graph_k']
    params['model_params']['lig_lig_edge_rbf_params'] = ligand_enc_params['model_params']['lig_lig_edge_rbf_params']

    # Load pretrained ligand encoder weights.
    fix_ligand_encoder_weights(model, checkpoint_state_dict, params['fix_ligand_encoder_weights'])

    if checkpoint_dict is not None:
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)

    # Wrap model with DDP to enable parallelization across devices.
    prev_steps_taken = 0
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    if checkpoint_dict is None:
        optimizer = get_std_opt(model.parameters(), params['model_params']['node_embedding_dim'], 0, warmup_steps=params['optimizer_warmup_steps'], factor=params['learning_rate_scale_factor'])
    else:
        optimizer_state_dict = checkpoint_dict['optimizer_state_dict']
        optimizer = get_std_opt(model.parameters(), params['model_params']['node_embedding_dim'], checkpoint_dict['optimizer_num_steps'], warmup_steps=params['optimizer_warmup_steps'], factor=params['learning_rate_scale_factor'])
        optimizer.optimizer.load_state_dict(optimizer_state_dict)
        optimizer._rate = optimizer.rate()
        prev_steps_taken = checkpoint_dict['optimizer_num_steps']

        assert np.isclose(optimizer.rate(), checkpoint_dict['optimizer_rate'], rtol=1e-5, atol=1e-5), f'Optimizer rate mismatch. {optimizer.rate()} != {checkpoint_dict["optimizer_rate"]}'

    # Watch the weight updates on the first device.
    if rank == 0 and params['use_wandb']:
        wandb.watch(model, log='all', log_freq=params['log_interval'])

    # Create the dataset object (this is the same between all devices).
    dataset = UnclusteredProteinChainDataset(params)
    # Define the collate function which will form batches from individual examples in dataset.
    collate_fn = lambda x: collate_sampler_data(x, params['use_xtal_additive_ligands'], recompute_all_cb_atoms=params['recompute_all_cb_atoms'], disable_ligand_information=params['disable_ligand_information'])
    amp_scaler = torch.amp.GradScaler() # type: ignore

    try:
        steps_taken = 0 + prev_steps_taken
        int_logs = defaultdict(int)
        list_logs = defaultdict(list)
        initial_log = False
        with tqdm(range(params['log_interval']), disable=(rank != 0), desc='Training', dynamic_ncols=True) as pbar:
            for epoch in range(starting_epoch_idx, params['num_epochs']):
                model.train()

                train_dataloader, test_dataloader = prepare_dataloaders(rank, world_size, params, epoch, dataset, collate_fn)

                for batch_idx, batch in enumerate(train_dataloader):
                    batch: BatchData

                    # Construct all batch tensors and perform node dropout regularization if necessary.
                    batch.to_device(device)
                    batch.sample_pseudoligands(params['num_residues_per_ligand'], params['min_contact_number_for_sampling'], model.module.rotamer_builder) # type: ignore
                    if params['recompute_all_cb_atoms']:
                        batch.recompute_cb_atoms()
                    batch.construct_graphs(
                        model.module.rotamer_builder,  # type: ignore
                        model.module.ligand_featurizer, # type: ignore
                        **params['model_params']['graph_structure'], 
                        protein_training_noise=params['protein_training_noise'], 
                        ligand_training_noise=params['ligand_training_noise'],
                        subgraph_only_dropout_rate=params['subgraph_only_dropout_rate'],
                        num_adjacent_residues_to_drop=params['sampled_residue_adjacent_drop_num'],
                        build_hydrogens=params['model_params']['build_hydrogens'],
                        use_aliphatic_ligand_hydrogens=params['use_aliphatic_ligand_hydrogens'],
                    )
                    batch.generate_decoding_order()

                    # Track which/number of residues we are actually training over in the current batch.
                    valid_residue_mask = (batch.sampled_chain_mask) & ((~batch.extra_atom_contact_mask) | (batch.sequence_indices == aa_short_to_idx['C']))
 
                    valid_first_shell_mask = batch.first_shell_ligand_contact_mask & valid_residue_mask
                    global_num_valid_residues = valid_residue_mask.sum()
                    local_num_valid_residues = global_num_valid_residues.clone()
                    if local_num_valid_residues.item() == 0:
                        continue

                    with autocast(device_type='cuda', dtype=torch.float16):
                        # Forward pass.    
                        sequence_logits, _, output_chi_logits, chi_offset_pred, true_chi_offsets = model(batch)

                        # Compute sequence loss.
                        seq_loss = compute_sequence_loss(sequence_logits, batch.sequence_indices, valid_residue_mask)

                        # Compute chi loss.
                        chi_mask = ~batch.chi_angles.isnan()
                        global_num_valid_chi = (~chi_mask[valid_residue_mask].isnan()).sum()
                        local_num_valid_chi = global_num_valid_chi.clone()
                        chi_encoding = model.module.rotamer_builder.compute_binned_degree_basis_function(batch.chi_angles).nan_to_num() # type: ignore
                        chi_loss = compute_chi_loss(output_chi_logits, chi_encoding, valid_residue_mask, chi_mask)
                        chi_offset_loss = params['chi_offset_loss_weight'] * compute_chi_offset_loss(chi_offset_pred, true_chi_offsets, valid_residue_mask)

                        # Compute MSA loss.
                        correct_sequence_mask = sequence_logits.argmax(dim=-1)[valid_residue_mask] == batch.sequence_indices[valid_residue_mask]
                        msa_loss = params['msa_loss_weight'] * compute_msa_loss(sequence_logits, batch.msa_data, valid_residue_mask, correct_sequence_mask)

                        # Sum valid residue count
                        dist.all_reduce(global_num_valid_residues)
                        dist.all_reduce(global_num_valid_chi)

                        # Scale losses by fraction in global batch size.
                        # https://github.com/pytorch/pytorch/issues/67253
                        seq_loss = seq_loss / global_num_valid_residues
                        msa_loss = msa_loss / global_num_valid_residues
                        chi_loss = chi_loss / global_num_valid_chi
                        chi_offset_loss = chi_offset_loss / global_num_valid_chi

                        # Scale loss by world size so when gradients are averaged by world_size on backwards, we get a larger effective batch size.
                        loss = world_size * (seq_loss + chi_loss + msa_loss + chi_offset_loss)

                    amp_scaler.scale(loss).backward() # type: ignore

                    # Compute gradients and update model.
                    # Adjust gradient magnitudes based on the number of accumulation steps.
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # type: ignore
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            list_logs['grad_norm'].append(param.grad.nan_to_num().abs().norm().item())

                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    steps_taken += 1

                    # Zero gradients.
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Update progress bar on master device.
                    pbar.update(1)

                    # Because the number of optimizer.step() calls isn't consistent between GPUs, we need to log 
                    # as a function of the number of steps taken by each GPU.
                    if steps_taken % params['log_interval'] == 0:

                        # Compute average loss and accuracy.
                        train_logs = {}
                        train_logs['loss'] = sum(list_logs['loss']) / (max(len(list_logs['loss']), 1) * world_size)
                        train_logs['seq_loss'] = sum(list_logs['seq_loss']) / max(len(list_logs['seq_loss']), 1)
                        train_logs['chi_loss'] = sum(list_logs['chi_loss']) / max(len(list_logs['chi_loss']), 1)
                        train_logs['msa_loss'] = sum(list_logs['msa_loss']) / max(len(list_logs['msa_loss']), 1)
                        train_logs['chi_offset_loss'] = sum(list_logs['chi_offset_loss']) / max(len(list_logs['chi_offset_loss']), 1)

                        train_logs['accuracy'] = int_logs['num_correct'] / max(int_logs['num_residues'], 1)
                        train_logs['fs_accuracy'] = int_logs['num_fs_correct'] / max(int_logs['num_fs_residues'], 1)

                        train_logs['grad_norm_mean'] = sum(list_logs['grad_norm']) / max(len(list_logs['grad_norm']), 1)
                        train_logs['grad_norm_max'] = max(list_logs['grad_norm'])
                        train_logs['grad_norm_min'] = min(list_logs['grad_norm'])

                        train_logs['rank'] = rank
                        train_logs['epoch'] = epoch

                        # Log test information when enough train log steps have occurred.
                        test_logs = {}
                        sampled_logs = {}
                        if (steps_taken % (params['train_logs_per_test_epoch'] * params['log_interval']) == 0) or not initial_log:
                            # # Test model, reset test dataloader, then sample the dataset.
                            # test_logs = test(rank, test_dataloader, model, params)
                            # test_dataloader = prepare_dataloaders(rank, world_size, params, epoch, dataset, collate_fn, test_only=True)
                            # sampled_logs = sample_dataset(rank, model, test_dataloader, epoch, params) # type: ignore
                            pass
                        all_logs = (train_logs, test_logs, sampled_logs)
        
                        # Gather all logs from all GPUs.
                        outputs = [None for _ in range(world_size)]
                        dist.all_gather_object(outputs, all_logs)
                        
                        # Log outputs on main GPU.
                        if rank == 0:
                            log_output = {}

                            # Aggregate log outputs.
                            log_output['optimizer_steps'] = steps_taken
                            log_output['epoch'] = epoch
                            log_output['lr'] = optimizer._rate
                            log_output['train'] = {
                                'loss': np.array([x[0]['loss'] for x in outputs]).sum(), # type: ignore
                                'chi_loss': np.array([x[0]['chi_loss'] for x in outputs]).sum(), # type: ignore
                                'seq_loss': np.array([x[0]['seq_loss'] for x in outputs]).sum(), # type: ignore
                                'msa_loss': np.array([x[0]['msa_loss'] for x in outputs]).sum(), # type: ignore
                                'chi_offset_loss': np.array([x[0]['chi_offset_loss'] for x in outputs]).sum(), # type: ignore
                                'accuracy': np.array([x[0]['accuracy'] for x in outputs]).mean(), # type: ignore
                                'fs_accuracy': np.array([x[0]['fs_accuracy'] for x in outputs]).mean(), # type: ignore
                                'grad_norm_mean': np.array([x[0]['grad_norm_mean'] for x in outputs]).mean(), # type: ignore
                                'grad_norm_max': np.array([x[0]['grad_norm_max'] for x in outputs]).mean(), # type: ignore
                                'grad_norm_min': np.array([x[0]['grad_norm_min'] for x in outputs]).mean(), # type: ignore
                            }

                            if (steps_taken % (params['train_logs_per_test_epoch'] * params['log_interval']) == 0) or not initial_log:
                                # log_output['test'] = {
                                #     'loss': np.array([x[1]['loss'] for x in outputs]).mean(), # type: ignore
                                #     'seq_loss': np.array([x[1]['seq_loss'] for x in outputs]).mean(), # type: ignore
                                #     'chi_loss': np.array([x[1]['chi_loss'] for x in outputs]).mean(), # type: ignore
                                #     'msa_loss': np.array([x[1]['msa_loss'] for x in outputs]).mean(), # type: ignore
                                #     'chi_offset_loss': np.array([x[1]['chi_offset_loss'] for x in outputs]).mean(), # type: ignore
                                #     'accuracy': np.array([x[1]['accuracy'] for x in outputs]).mean(), # type: ignore
                                #     'fs_accuracy': np.array([x[1]['fs_accuracy'] for x in outputs]).mean(), # type: ignore

                                #     'sampled_sequence_recovery': np.array([x[2]['sampled_sequence_recovery'] for x in outputs]).mean(), # type: ignore
                                #     'sampled_first_shell_sequence_recovery': np.array([x[2]['sampled_first_shell_sequence_recovery'] for x in outputs]).mean(), # type: ignore
                                #     'sampled_rmsd': average_between_dicts([x[2]['sampled_rmsd'] for x in outputs]), # type: ignore
                                #     'sampled_chi_accuracy': average_between_dicts([x[2]['sampled_chi_accuracy'] for x in outputs]), # type: ignore
                                # }
                                save_model_checkpoint(model, params, ligand_enc_params, optimizer, steps_taken, epoch)

                            # Log to wandb if necessary.
                            if params['use_wandb']:
                                wandb.log(log_output)
                            pbar.write(repr(log_output))

                            # Reset log counters.
                            int_logs = defaultdict(int)
                            list_logs = defaultdict(list)

                        # Finish training.
                        if max([x[0]['epoch'] if x is not None else torch.nan for x in outputs]) >= params['num_epochs']:
                            raise TrainingFinishedException

                        # After first log, only log on intervals of train_logs_per_test_epoch.
                        initial_log = True

                        # reset enabled progress bars for new log delay interval. 
                        pbar.reset()

                    # Track loss and accuracy.
                    num_correct = (sequence_logits[valid_residue_mask].argmax(dim=-1) == batch.sequence_indices[valid_residue_mask]).sum().item()
                    num_residues = int(valid_residue_mask.sum().item())
                    num_fs_correct = (sequence_logits[valid_first_shell_mask].argmax(dim=-1) == batch.sequence_indices[valid_first_shell_mask]).sum().item()
                    num_fs_residues = int(valid_first_shell_mask.sum().item())

                    list_logs['loss'].append(loss.item())
                    list_logs['seq_loss'].append(seq_loss.item())
                    list_logs['chi_loss'].append(chi_loss.item())
                    list_logs['msa_loss'].append(msa_loss.item())
                    list_logs['chi_offset_loss'].append(chi_offset_loss.item())
                    int_logs['num_correct'] += num_correct
                    int_logs['num_residues'] += num_residues
                    int_logs['num_fs_correct'] += num_fs_correct
                    int_logs['num_fs_residues'] += num_fs_residues
    except Exception:
        print(traceback.format_exc())
        print("Quitting all processes...")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":

    # Use these to train the LASErMPNN encoder without pretrained weights.
    default_ligand_encoder_params = {
        'model_params': {
            'num_ligand_encoder_vectors': 15, 'node_embedding_dim': 256, 'ligand_edge_embedding_dim': 128, 
            'atten_dimension_upscale_factor': None,
            'lig_lig_edge_rbf_params': {
                'num_bins': 75, 'bin_min': 0.0, 'bin_max': 15.0,
            }, 
            'graph_structure': { 'lig_lig_knn_graph_k': 5, },
            'num_encoder_layers': 3,
            'num_attention_heads': 3,
            'dropout': 0.1,
            'atten_head_aggr_layers': 0,
        }
    }

    ALTERNATE_DATABASE_PATH = Path('/home/nickhedu/programs/LASErMPNN/')
    params = {
        'debug': (debug := False),
        'use_wandb': True and not debug,
        'use_data_augmentations': (augment := True),
        'random_seed': None if not debug else 42,
        'master_port': '12997',

        # We attempted to create a 'soluble-mpnn' version of lasermpnn but the sequences it generates look qualitatively worse than the standard lasermpnn model.
        'soluble_proteins_only': True,

        # Idealizes all residue frames if set to True, not ideal for foldability of natural proteins, but may be more useful for de novo design.
        'recompute_all_cb_atoms': True, 

        # If set to False, noise is applied at the frame-level rather than per backbone atom, False improves dihedral angle decoding at expense of sequence foldability.
        'use_per_atom_backbone_noise': False,

        # If True, trains a "ProteinMPNN-like" model by not loading any heteroatom information.
        'disable_ligand_information': False,
        'devices': VISIBLE_DEVICES,

        'checkpoint_to_resume_from': None,
        'output_weights_checkpoint_prefix': CURR_FILE_DIR_PATH / 'model_weights/jerry_weights',
        
        'pretrained_ligand_encoder_weights': CURR_FILE_DIR_PATH / 'model_weights/pretrained_ligand_encoder_weights.pt',
        # 'pretrained_ligand_encoder_weights': default_ligand_encoder_params, # Uncomment to disable pretrained ligand encoder.

        # 'raw_dataset_path': CURR_FILE_DIR_PATH / 'databases/pdb_dataset/dataset_shelve',
        # 'metadata_dataset_path': CURR_FILE_DIR_PATH / 'databases/pdb_dataset/metadata_shelve',
        # 'clustering_dataframe_path': CURR_FILE_DIR_PATH / 'databases/pdb_dataset/cluster_representative_add_bromo.pkl',
        # 'subcluster_pickle_path': CURR_FILE_DIR_PATH / 'databases/pdb_dataset/subcluster_pickle.pkl',
        'raw_dataset_path': ALTERNATE_DATABASE_PATH / 'databases/pdb_dataset/dataset_shelve',
        'metadata_dataset_path': ALTERNATE_DATABASE_PATH / 'databases/pdb_dataset/metadata_shelve',
        'clustering_dataframe_path': ALTERNATE_DATABASE_PATH / 'databases/pdb_dataset/cluster_representative_add_bromo.pkl',
        'subcluster_pickle_path': ALTERNATE_DATABASE_PATH / 'databases/pdb_dataset/subcluster_pickle.pkl',

        'num_epochs': 500,
        'batch_size': 6_000, # Practically, this is the batch size per GPU... so 5_000 * num_devices = total batch size.
        'log_interval': 100 if not debug else 5,
        'learning_rate_scale_factor': 2,
        'optimizer_warmup_steps': 5000,
        'num_validation_samples': 100 if not debug else 5,
        'train_logs_per_test_epoch': 50,

        'fix_ligand_encoder_weights': True,
        'use_aliphatic_ligand_hydrogens': False,
        'use_xtal_additive_ligands': True,
        'num_residues_per_ligand': 150,

        'switch_to_rmsd_sidechain_loss_epoch': torch.inf,
        'min_contact_number_for_sampling': 3 if augment else torch.inf,
        'sampled_residue_adjacent_drop_num': 6,
        'bias_sample_to_proteins_with_ligand_fraction': 0.75,
        'min_num_ligand_contacting_residues_for_bias_sample': 5,
        'max_protein_size': 6000 if not debug else 500,
        'msa_loss_weight': 0.1,
        'chi_offset_loss_weight': 0.5,

        'protein_training_noise': 0.1 if augment else 0.0,
        'ligand_training_noise': 0.05 if augment else 0.0,
        'subgraph_only_dropout_rate': -1 if augment else 0.0,

        'sample_randomly': True,
        'autoregressive_test_batch_sample_fraction': 1.0 if not debug else 0.05,
        'model_params': {
            'build_hydrogens': True,
            'additional_ligand_mlp': True,
            'num_laser_vectors': 10,
            'node_embedding_dim': 256,
            'protein_edge_embedding_dim': 128,
            'ligand_edge_embedding_dim': 128,
            'dropout': 0.1,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'chi_angle_rbf_bin_width': 5,
            'atten_dimension_upscale_factor': 4,
            'num_attention_heads': 1,
            'atten_head_aggr_layers': 0,
            'graph_structure': {
                'pr_pr_knn_graph_k': 48,
                'lig_pr_distance_cutoff': 20.0,
                'lig_pr_knn_graph_k': 48,
            },
            'prot_prot_edge_rbf_params': {
                'num_bins': 16,
                'bin_min': 2,
                'bin_max': 22,
            },
            'lig_prot_edge_rbf_params': {
                'num_bins': 75,
                'bin_min': 0.0,
                'bin_max': 15.0,
            },
        },
    }

    world_size = len(VISIBLE_DEVICES)
    mp.spawn(train, args=(world_size, params), nprocs=world_size) # type: ignore

