import os
import shelve
import numpy as np
import prody as pr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch_cluster import knn_graph
from torch_scatter import scatter
from torch.utils.data import Dataset, Sampler

from .constants import MAX_PEPTIDE_LENGTH, MIN_TM_SCORE_FOR_SIMILARITY, HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD, COVALENT_HYDROGEN_BOND_MAX_DISTANCE, aa_short_to_idx, aa_idx_to_short, atom_to_atomic_number, aa_to_chi_angle_mask, ligandmpnn_validation_pdb_codes, ligandmpnn_training_pdb_codes, ligandmpnn_test_pdb_codes, ideal_prot_aa_coords, hydrogen_extended_dataset_atom_order
from .build_rotamers import RotamerBuilder, compute_alignment_matrices, apply_transformation, extend_coordinates
from .ligand_featurization import LigandFeaturizer
from .hbond_network import RigorousHydrogenBondNetworkDetector, compute_hbonding_connected_component
from .helper_functions import create_prody_protein_from_coordinate_matrix
from dataclasses import dataclass
from pykeops.torch import LazyTensor


@dataclass
class SampledSidechainData():
    """
    Dataclass storing the data we can use for sidechain-sampling-as-ligands for a batch of proteins.
    """
    sampled_indices: torch.Tensor
    sampled_batch_indices: torch.Tensor
    sampled_subbatch_indices: torch.Tensor
    batch_idx_to_num_clusters: Dict[int, int]
    

@dataclass
class UnprocessedLigandData():
    lig_atomic_numbers: torch.Tensor
    lig_coords: torch.Tensor
    lig_batch_indices: torch.Tensor
    lig_subbatch_indices: torch.Tensor

    def extend(self, other: 'UnprocessedLigandData') -> 'UnprocessedLigandData':
        """
        Appends another UnprocessedLigandData object to the current one.
        """

        return UnprocessedLigandData(
            torch.cat([self.lig_atomic_numbers, other.lig_atomic_numbers], dim=0),
            torch.cat([self.lig_coords, other.lig_coords], dim=0),
            torch.cat([self.lig_batch_indices, other.lig_batch_indices], dim=0),
            torch.cat([self.lig_subbatch_indices, other.lig_subbatch_indices], dim=0),
        )
    
    def to_device(self, device):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
    
    def pin_memory(self) -> 'UnprocessedLigandData':
        """
        Pins all tensors in the BatchData object to memory.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.pin_memory()
        return self


@dataclass
class LigandData():
    """
    Dataclass storing ligand data which can be a component of BatchData.
    """
    lig_nodes: torch.Tensor
    lig_coords: torch.Tensor
    lig_batch_indices: torch.Tensor
    lig_subbatch_indices: torch.Tensor
    lig_lig_edge_index: Optional[torch.Tensor] = None
    lig_lig_edge_distance: Optional[torch.Tensor] = None
    real_ligand_mask: Optional[torch.Tensor] = None
    ligand_atomic_numbers: Optional[torch.Tensor] = None

    def to_device(self, device):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
    
    def pin_memory(self) -> 'LigandData':
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.pin_memory()
        return self


@dataclass
class BatchData():
    """
    Dataclass storing all the data for a batch of proteins for input to model.

    Attributes:
        pdb_codes, sequence_indices, chi_angles, backbone_coords,
        residue_burial_counts, batch_indices, chain_indices, resnum_indices,
        chain_mask, extra_atom_contact_mask, msa_data, msa_depth_weight,
        sampled_pseudoligands, lig_nodes, lig_coords, pr_pr_edge_index,
        pr_pr_edge_distance, lig_pr_edge_index, lig_pr_edge_distance,
        lig_lig_edge_index, lig_lig_edge_distance, decoding_order

    Properties:
        device, num_residues

    Methods: 
        sample_pseudoligands
        construct_graphs
        generate_decoding_order
        to_device
    """

    # TODO: extract the stuff that isn't necessary to pass into model to a TrainingMetadata class.
    pdb_codes: List[str]

    sequence_indices: torch.Tensor
    chi_angles: torch.Tensor
    backbone_coords: torch.Tensor
    phi_psi_angles: torch.Tensor # Helps build hydrogens on backbone nitrogen & with placing methyl cap coordinates.
    sidechain_contact_number: torch.Tensor # Precomputed number of other residues not adj. in sequence in contact with current residue's sidechain.
    residue_burial_counts: torch.Tensor # Tracks number of CB atoms within 10A of a given residue's CB atom.
    sc_mediated_hbond_counts: torch.Tensor # Tracks number of hydrogen bonds contacting each residue/backbone.

    # Tracks which protein in batch each residue belongs to.
    batch_indices: torch.Tensor
    # Tracks which chain in the protein each residue belongs to.
    chain_indices: torch.Tensor
    # Tracks the positional index of the residue in the protein according to crystallized sequence.
    resnum_indices: torch.Tensor
    # True if residue is NOT being trained over.
    chain_mask: torch.Tensor 
    sampled_chain_mask: torch.Tensor
    # True if residue is in contact with extra atoms.
    extra_atom_contact_mask: torch.Tensor 
    first_shell_ligand_contact_mask: torch.Tensor

    msa_data: torch.Tensor
    msa_depth_weight: torch.Tensor

    # Data to store ligands parsed from the PDB.
    unprocessed_ligand_input_data: Optional[UnprocessedLigandData] = None
    sampled_pseudoligands: Optional[SampledSidechainData] = None
    ligand_data: Optional[LigandData] = None

    # Edge index and distance tensors:
    pr_pr_edge_index: Optional[torch.Tensor] = None
    pr_pr_edge_distance: Optional[torch.Tensor] = None
    lig_pr_edge_index: Optional[torch.Tensor] = None
    lig_pr_edge_distance: Optional[torch.Tensor] = None

    decoding_order: Optional[torch.Tensor] = None

    def to_device(self, device: torch.device) -> None:
        """
        Moves all tensors in the BatchData object to the specified device.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
            if isinstance(v, UnprocessedLigandData):
                self.__dict__[k].to_device(device)
            if isinstance(v, LigandData):
                self.__dict__[k].to_device(device)
        
    def pin_memory(self) -> 'BatchData':
        """
        Pins all tensors in the BatchData object to memory.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.pin_memory()
            if isinstance(v, UnprocessedLigandData):
                self.__dict__[k] = v.pin_memory()
            if isinstance(v, LigandData):
                self.__dict__[k] = v.pin_memory()

        raise NotImplementedError("This seems to create torch memory errors when cpu-gpu synchronization occurs, might be fixed in newer versions of pytorch but don't use pin_memory=True in dataloader for now.")
        return self

    @property
    def device(self) -> torch.device:
        """
        Returns the device that the batch tensors are currently on when addressed as model.device
        """
        return self.backbone_coords.device
    
    @property
    def num_residues(self) -> int:
        return self.backbone_coords.shape[0]

    def recompute_cb_atoms(self) -> None:
        """
        There is signal in the crystallographic frame coordinates that we can ablate by 
            idealizing the backbone frame and imputing virtual CB coords. This has already been done to compute a 
            'virtual_cb_adj_chi_angles' tensor in the dataset which is the chi angles relative to the ideal frame atoms 
            and not the crystallographic atoms.
        """
        self.backbone_coords = idealize_backbone_coords(self.backbone_coords, self.phi_psi_angles)

    def sample_pseudoligands(self, num_residues_per_ligand: int, min_contact_number_for_sampling: int, rotamer_builder: RotamerBuilder) -> None:
        """
        Given a batch of proteins, samples a single residue from each protein structural cluster (not in contact with a real ligand) to use as a ligand.

        Adds SampledSidechainData object to self.sampled_pseudoligands which stores the indices into 
            protein metadata tensors indices and the batch indices they came from.
        
        num_residues_per_ligand (int):
            Num residues in a protein complex before we sample an extra ligand. 
            If len(protein) > 2*num_residues_per_ligand, samples two sidechains ligands by kmeans clustering of the 
                CA atoms and sampling one residue from each cluster.
        """

        # Ensure that we have the expected number of chi angles for the residues were sampling.
        xremoved_idces = self.sequence_indices.clone()
        is_x_mask = self.sequence_indices == aa_short_to_idx['X']
        xremoved_idces[is_x_mask] = aa_short_to_idx['G']

        # Identify residues that are buried enough to be sampled.
        # Constrains sampling to pseudoligands with at least one sidechain-mediated hydrogen bond.
        is_sampleable_mask = (self.sidechain_contact_number >= min_contact_number_for_sampling) & (self.sc_mediated_hbond_counts > 0)

        # Identify residues that are in the chains we are training over and sampleable.
        valid_to_sample_mask = (~self.extra_atom_contact_mask) & (~self.chain_mask) & is_sampleable_mask & (~is_x_mask)
        valid_to_sample_idx = valid_to_sample_mask.nonzero().flatten()

        # Identify the proteins in batch that have valid residues to sample and what those residues are.
        valid_batch_members = self.batch_indices[valid_to_sample_mask]
        unique_batch_indices = torch.unique(valid_batch_members, sorted=True)
        valid_residue_indices = self.sequence_indices[valid_to_sample_mask]
        valid_ca_coords = self.backbone_coords[valid_to_sample_mask][:, 1]
        valid_first_shell_mask = self.first_shell_ligand_contact_mask[valid_to_sample_mask]

        # Create a list of indicies into the batch dimension from which to sample as ligands.
        # sampled_indices = torch.zeros((unique_batch_indices.shape[0],), dtype=torch.long, device=self.chain_mask.device)
        batch_idx_to_num_clusters = {}
        sampled_indices, sampled_batch_indices, sampled_subbatch_indices = [], [], []
        for idx in range(unique_batch_indices.shape[0]):
            # Handle a single protein in the batch at a time.
            ###  Get the index of the protein in within the batch to sample.
            protein_idx = unique_batch_indices[idx]

            ### Mask for just the valid residues to sample in the current protein.
            valid_residues_curr_complex_mask = valid_batch_members == protein_idx

            ### Mask for the current protein in the batch.
            num_clusters = max(valid_residues_curr_complex_mask.sum().item() // num_residues_per_ligand, 1)
            curr_ca_coords = valid_ca_coords[valid_residues_curr_complex_mask]
            if num_clusters < 2:
                clusters = torch.zeros((curr_ca_coords.shape[0],), dtype=torch.long, device=self.device)
            else:
                clusters, _ = compute_kmeans_clusters(curr_ca_coords.contiguous(), k=num_clusters)

            first_shell_contacting_clusters = torch.unique(clusters[valid_first_shell_mask[valid_residues_curr_complex_mask]])

            # Sample once from each cluster if possible.
            num_sampled = 0
            for cluster_idx in range(num_clusters):

                if cluster_idx in first_shell_contacting_clusters:
                    continue

                # Get the indices of the residues in the current cluster.
                curr_cluster_mask = clusters == cluster_idx
                curr_sample_indices = valid_residues_curr_complex_mask.nonzero().flatten()[curr_cluster_mask]

                # Get the amino acid types we can sample for this protein.
                sampleable_residues = valid_residue_indices[curr_sample_indices] 
                sampleable_residue_types = sampleable_residues.unique()
                if sampleable_residue_types.shape[0] == 0:
                    continue

                # Uniformly sample a residue type (to not over-sample more common residues)
                sampled_residue_type = sampleable_residue_types[torch.randint(low=0, high=sampleable_residue_types.shape[0], size=(1, ))]

                # Get the indices (into protein_mask) of the residues of the sampled type.
                is_sampled_residue_mask = sampleable_residues == sampled_residue_type

                ### Convert sampled residue mask to tensor of indices into protein_mask.
                protein_mask_indices = curr_sample_indices[is_sampled_residue_mask]
                ### Convert protein_mask_indices to indices into indices into valid to sample mask/original batch tensors.
                valid_to_sample_indices = valid_to_sample_idx[protein_mask_indices]

                # Randomly sample one of the indices that correspond to the type of residue we can sample and record it.
                sampled_residue_index = valid_to_sample_indices[torch.randint(low=0, high=valid_to_sample_indices.shape[0], size=(1, ))]
                sampled_indices.append(sampled_residue_index)
                sampled_batch_indices.append(protein_idx.unsqueeze(-1))
                sampled_subbatch_indices.append(torch.tensor([num_sampled], dtype=torch.long, device=self.device))
                num_sampled += 1

            # Not all clusters are guaranted to have a valid residue to sample so we need to track how many we actually sample.
            batch_idx_to_num_clusters[protein_idx.item()] = num_sampled

        self.sampled_pseudoligands = SampledSidechainData(
            torch.cat(sampled_indices, dim=0) if len(sampled_indices) > 0 else torch.empty((0,), dtype=torch.long, device=self.device),
            torch.cat(sampled_batch_indices, dim=0) if len(sampled_batch_indices) > 0 else torch.empty((0,), dtype=torch.long, device=self.device),
            torch.cat(sampled_subbatch_indices, dim=0) if len(sampled_subbatch_indices) > 0 else torch.empty((0,), dtype=torch.long, device=self.device),
            batch_idx_to_num_clusters,
        )

    def _generate_pseudoligand_coords(
            self, batch_idx, unique_subbatch_indices, curr_sampled_indices, curr_batch_prot_mask, 
            pseudolig_coords, pseudolig_residue_sequence_indices, pseudolig_nodes, pseudolig_atomic_numbers, batch_index_tensor, subbatch_index_tensor, 
            rotamer_builder, ligand_featurizer, num_adjacent_residues_to_drop
    ):
        """
        Given pseudoligands sampled for the current batch, generates ligand featurization and coordinates for the sampled residues and builds methylamine caps on the N and C termini.
        """

        pseudoligand_coords = pseudolig_coords[unique_subbatch_indices]
        subbatch_sequence_indices = pseudolig_residue_sequence_indices[unique_subbatch_indices]

        subbatch_polymer_indices = self.resnum_indices[curr_sampled_indices]
        subbatch_chain_indices = self.chain_indices[curr_sampled_indices]
        pseudolig_phi_psi = self.phi_psi_angles[curr_sampled_indices]

        protein_complex_resnum_indices = self.resnum_indices[curr_batch_prot_mask]
        protein_complex_chain_indices = self.chain_indices[curr_batch_prot_mask]

        output_residue_drop_mask = torch.zeros_like(protein_complex_chain_indices, dtype=torch.bool)
        output_ligand_nodes, output_ligand_coords, output_ligand_batch_indices, output_ligand_subbatch_indices, output_atomic_numbers = [], [], [], [], []
        for idx, subbatch_idx in enumerate(unique_subbatch_indices):

            # Get relevant metadata for subbatch ligand.
            subbatch_mask = subbatch_index_tensor == subbatch_idx
            subbatch_nodes = pseudolig_nodes[subbatch_mask]
            subbatch_atomic_numbers = pseudolig_atomic_numbers[subbatch_mask]
            subbatch_batch_idx_tensor = batch_index_tensor[subbatch_mask] # Tensor of subbatch indices in shape of ligand coords.
            subbatch_subbatch_idx_tensor = subbatch_index_tensor[subbatch_mask] # Tensor of subbatch indices in shape of ligand coords.

            # Get the coordinates and phi/psi angles for the subbatch ligand.
            premod_coords = pseudoligand_coords[idx].clone()
            subbatch_coords = pseudoligand_coords[idx]
            subbatch_phi_psi = pseudolig_phi_psi[idx]
            subbatch_sequence_idx = subbatch_sequence_indices[idx] # The amino acid identity index.
            subbatch_chain_idx = subbatch_chain_indices[idx] # The chain index the pseudoligand is in.
            subbatch_polymer_sequence_idx = subbatch_polymer_indices[idx] # The polymer sequence index.

            # Compute cap and indices for the subbatch ligand.
            cap_coords, cap_atomic_numbers = rotamer_builder._compute_methyl_cap_coordinates(subbatch_phi_psi, subbatch_coords, subbatch_sequence_idx)
            cap_ligand_node_features = ligand_featurizer.encode_ligand_from_atomic_number(cap_atomic_numbers)
            cap_batch_indices = torch.full((cap_coords.shape[0],), batch_idx, dtype=torch.long, device=cap_coords.device)
            cap_subbatch_indices = torch.full((cap_coords.shape[0],), subbatch_idx, dtype=torch.long, device=cap_coords.device)

            # Cys and His residues can be deprotonated, so we need to track which atoms are deprotonated.
            missing_hydrogen_mask = torch.zeros((subbatch_coords.shape[0],), dtype=torch.bool, device=subbatch_coords.device)
            if subbatch_sequence_idx.item() == aa_short_to_idx['C']:
                missing_hydrogen_mask = rotamer_builder.cys_hydrogen_optional_mask & subbatch_coords.isnan().any(dim=-1)
            if subbatch_sequence_idx.item() == aa_short_to_idx['H']:
                missing_hydrogen_mask = rotamer_builder.his_hydrogen_optional_mask & subbatch_coords.isnan().any(dim=-1)

            # Drop out NaN padding coordinates from the sampled amino acid.
            sampled_aa_mask = (~subbatch_coords.isnan()).all(dim=-1)
            subbatch_coords = subbatch_coords[sampled_aa_mask]
            missing_hydrogen_mask = missing_hydrogen_mask[sampled_aa_mask | missing_hydrogen_mask] # Drop padding from missing hydrogen mask.

            # Drop features for missing atoms.
            assert subbatch_nodes.shape[0] == missing_hydrogen_mask.shape[0], f"Mismatched shapes for ligand nodes and missing hydrogen mask. {subbatch_nodes.shape=}, {missing_hydrogen_mask.shape=}. Sampled {aa_idx_to_short[subbatch_sequence_idx.item()]=}, {(~premod_coords.isnan()).any(dim=-1).sum()}, "
            subbatch_nodes = subbatch_nodes[~missing_hydrogen_mask]
            subbatch_atomic_numbers = subbatch_atomic_numbers[~missing_hydrogen_mask]
            subbatch_batch_idx_tensor = subbatch_batch_idx_tensor[~missing_hydrogen_mask]
            subbatch_subbatch_idx_tensor = subbatch_subbatch_idx_tensor[~missing_hydrogen_mask]

            # Merge the sampled amino acid coordinates and methyl cap coordinates and ligand features.
            curr_lig_nodes = torch.cat([subbatch_nodes, cap_ligand_node_features], dim=0)
            curr_lig_atomic_numbers = torch.cat([subbatch_atomic_numbers, cap_atomic_numbers - 1], dim=0)
            subbatch_coords = torch.cat([subbatch_coords, cap_coords], dim=0)
            subbatch_batch_idx_tensor = torch.cat([subbatch_batch_idx_tensor, cap_batch_indices], dim=0)
            subbatch_subbatch_idx_tensor = torch.cat([subbatch_subbatch_idx_tensor, cap_subbatch_indices], dim=0)
            assert curr_lig_nodes.shape[0] == subbatch_coords.shape[0], f"Mismatched shapes for ligand nodes and coordinates. {curr_lig_nodes.shape=}, {subbatch_coords.shape=}"
            assert curr_lig_atomic_numbers.shape[0] == subbatch_coords.shape[0], f"Mismatched shapes for ligand atomic numbers and coordinates. {curr_lig_atomic_numbers.shape=}, {subbatch_coords.shape=}"
            assert subbatch_batch_idx_tensor.shape[0] == subbatch_coords.shape[0], f"Mismatched shapes for ligand batch indices and coordinates. {subbatch_batch_idx_tensor.shape=}, {subbatch_coords.shape=}"
            assert subbatch_subbatch_idx_tensor.shape[0] == subbatch_coords.shape[0], f"Mismatched shapes for ligand batch indices and coordinates. {subbatch_subbatch_idx_tensor.shape=}, {subbatch_coords.shape=}"

            # Identify residues to drop from the protein complex and track final ligand atom metadata.
            same_chain_mask = protein_complex_chain_indices == subbatch_chain_idx
            drop_residues_mask = (torch.abs(protein_complex_resnum_indices - subbatch_polymer_sequence_idx) <= num_adjacent_residues_to_drop) & same_chain_mask
            output_residue_drop_mask |= drop_residues_mask
            output_ligand_nodes.append(curr_lig_nodes)
            output_atomic_numbers.append(curr_lig_atomic_numbers)
            output_ligand_coords.append(subbatch_coords)
            output_ligand_batch_indices.append(subbatch_batch_idx_tensor)
            output_ligand_subbatch_indices.append(subbatch_subbatch_idx_tensor)
        
        return torch.cat(output_ligand_coords, dim=0), torch.cat(output_ligand_nodes, dim=0), torch.cat(output_atomic_numbers, dim=0), torch.cat(output_ligand_batch_indices, dim=0), torch.cat(output_ligand_subbatch_indices, dim=0), output_residue_drop_mask

    def construct_graphs(
            self, rotamer_builder: RotamerBuilder, ligand_featurizer: LigandFeaturizer, pr_pr_knn_graph_k: int, 
            lig_pr_distance_cutoff: float, lig_pr_knn_graph_k: int, lig_lig_knn_graph_k: int, 
            protein_training_noise: float, ligand_training_noise: float, 
            subgraph_only_dropout_rate: float = 0.0, num_adjacent_residues_to_drop: int = 0, build_hydrogens: bool = True, 
            use_aliphatic_ligand_hydrogens: bool = True
    ) -> None:
        """
        Computes a KNN graph using CA coordinates and distances between all pairs of atoms.
        Stores the edge_index and edge_distance tensors in the BatchData object.

        If sample_pseudoligands has been called, computes the ligand-protein and ligand-ligand graphs using those residues as pseudo-ligands, 
            drops out all residues that are within +/- num_adjacent_residues_to_drop residues of the sampled residues and on the same chain from all tensors specified in
            `_apply_mask_to_residue_metadata_tensors` including backbone coordinates.
        """

        # Noise the backbone coordinates with x,y,z noise for each frame independently.
        if protein_training_noise > 0.0:
            noised_backbone_coords = torch.round(self.backbone_coords, decimals=2) + (protein_training_noise * torch.randn((self.backbone_coords.shape[0], 1, 3), device=self.device))
            self.backbone_coords = noised_backbone_coords

        # Identify the batch indices that have real ligands.
        if self.unprocessed_ligand_input_data is not None:
            real_ligand_batch_indices = self.unprocessed_ligand_input_data.lig_batch_indices
        else: 
            real_ligand_batch_indices = torch.empty((0,), dtype=torch.long, device=self.device)

        # Precompute pseudoligand coordinates, node features, and indices.
        pseudoligand_coords, pseudoligand_batch_indices, pseudoligand_subbatch_indices, pseudoligand_lig_nodes = None, None, None, None
        if self.sampled_pseudoligands is not None:
            pseudoligand_coords = rotamer_builder.build_rotamers(
                self.backbone_coords[self.sampled_pseudoligands.sampled_indices], 
                self.chi_angles[self.sampled_pseudoligands.sampled_indices], 
                self.sequence_indices[self.sampled_pseudoligands.sampled_indices],
                add_nonrotatable_hydrogens = build_hydrogens
            )
            assert isinstance(pseudoligand_coords, torch.Tensor) # Appease type checker.

            sampled_sequence_indices = self.sequence_indices[self.sampled_pseudoligands.sampled_indices]
            pseudoligand_lig_nodes, pseudoligand_batch_indices, pseudoligand_subbatch_indices, pseudoligand_atomic_numbers = ligand_featurizer.generate_ligand_nodes_from_amino_acid_labels(sampled_sequence_indices, self.sampled_pseudoligands.sampled_batch_indices, self.sampled_pseudoligands.sampled_subbatch_indices)

        prot_index_offset, lig_index_offset = 0, 0
        all_lig_coords, all_prot_prot_eidces, all_lig_prot_eidces, all_lig_lig_eidces, all_lig_nodes, all_lig_batch_indices, all_lig_subbatch_indices, all_lig_node_masks, first_shell_masks, all_lig_atomic_numbers = [], [], [], [], [], [], [], [], [], []
        full_batch_node_mask = torch.ones((self.num_residues,), dtype=torch.bool, device=self.device)
        for batch_idx in range(self.batch_indices.max() + 1):
            # Mask for the current batch.
            curr_batch_prot_mask = self.batch_indices == batch_idx
            residues_to_drop_mask = torch.zeros(int(curr_batch_prot_mask.sum()), dtype=torch.bool, device=self.device)
            curr_first_shell_mask = torch.zeros(int(curr_batch_prot_mask.sum()), dtype=torch.bool, device=self.device)
            real_ligand_node_mask = real_ligand_batch_indices == batch_idx
            curr_pseudoligand_batch_mask = pseudoligand_batch_indices == batch_idx
            curr_complex_backbone_coords = self.backbone_coords[curr_batch_prot_mask]

            if (real_ligand_node_mask.numel() == 0) or ((not real_ligand_node_mask.any().item()) and (not curr_pseudoligand_batch_mask.any().item())):
                curr_prot_prot_eidx = knn_graph(curr_complex_backbone_coords[:, 1], k=pr_pr_knn_graph_k, loop=True) + prot_index_offset
                all_prot_prot_eidces.append(curr_prot_prot_eidx)
                first_shell_masks.append(curr_first_shell_mask)
                prot_index_offset += curr_complex_backbone_coords.shape[0]
                continue

            # Handle real ligand graph generation.
            # TODO: defer lig_node computation if not using only atomic number featurizations.
            curr_lig_coords, curr_lig_nodes, curr_batch_idces, curr_subbatch_idces, curr_lig_atomic_numbers, curr_real_ligand_mask = None, None, None, None, None, None
            if real_ligand_node_mask.any().item():
                assert self.unprocessed_ligand_input_data is not None, "Unreachable."
                curr_lig_coords = self.unprocessed_ligand_input_data.lig_coords[real_ligand_node_mask]
                curr_lig_atomic_numbers = self.unprocessed_ligand_input_data.lig_atomic_numbers[real_ligand_node_mask]
                curr_lig_nodes = ligand_featurizer.encode_ligand_from_atomic_number(curr_lig_atomic_numbers)
                curr_lig_atomic_numbers = curr_lig_atomic_numbers - 1 # Adjust number to index.
                curr_batch_idces = self.unprocessed_ligand_input_data.lig_batch_indices[real_ligand_node_mask]
                curr_subbatch_idces = self.unprocessed_ligand_input_data.lig_subbatch_indices[real_ligand_node_mask]
                curr_real_ligand_mask = torch.ones((curr_lig_coords.shape[0],), dtype=torch.bool, device=self.device)

            # Handle pseudoligand graph generation.
            if self.sampled_pseudoligands is not None and curr_pseudoligand_batch_mask.any().item():
                assert pseudoligand_coords is not None and pseudoligand_batch_indices is not None and pseudoligand_subbatch_indices is not None and pseudoligand_lig_nodes is not None, "Unreachable."

                # Get the indices of the pseudoligands in the current batch.
                batch_index_pseudoligand_mask = self.sampled_pseudoligands.sampled_batch_indices == batch_idx
                curr_subbatch_indices = self.sampled_pseudoligands.sampled_subbatch_indices[batch_index_pseudoligand_mask]
                curr_sampled_indices = self.sampled_pseudoligands.sampled_indices[batch_index_pseudoligand_mask]
                curr_pseudoligand_coords = pseudoligand_coords[batch_index_pseudoligand_mask]
                curr_sequence_indices = sampled_sequence_indices[batch_index_pseudoligand_mask]

                # Generate the pseudoligand nodes and coordinates.
                pseudolig_coords, pseudolig_nodes, pseudolig_atomic_numbers, pseudolig_batch_idces, pseudolig_subbatch_idces, residues_to_drop_mask = self._generate_pseudoligand_coords(
                    batch_idx, curr_subbatch_indices, curr_sampled_indices, curr_batch_prot_mask,
                    curr_pseudoligand_coords.float(), curr_sequence_indices,
                    pseudoligand_lig_nodes[curr_pseudoligand_batch_mask],
                    pseudoligand_atomic_numbers[curr_pseudoligand_batch_mask],
                    pseudoligand_batch_indices[curr_pseudoligand_batch_mask],
                    pseudoligand_subbatch_indices[curr_pseudoligand_batch_mask],
                    rotamer_builder, ligand_featurizer, num_adjacent_residues_to_drop
                )
                pseudoligand_node_mask = torch.zeros((pseudolig_coords.shape[0],), dtype=torch.bool, device=self.device)

                # Merge pseudoligands with real ligands if they exist.
                if real_ligand_node_mask.any().item():
                    assert curr_lig_coords is not None and curr_lig_nodes is not None and curr_batch_idces is not None and curr_subbatch_idces is not None and curr_lig_atomic_numbers is not None and curr_real_ligand_mask is not None, "Unreachable."
                    curr_lig_coords = torch.cat([curr_lig_coords, pseudolig_coords], dim=0)
                    curr_lig_nodes = torch.cat([curr_lig_nodes, pseudolig_nodes], dim=0)
                    curr_lig_atomic_numbers = torch.cat([curr_lig_atomic_numbers, pseudolig_atomic_numbers], dim=0)
                    curr_batch_idces = torch.cat([curr_batch_idces, pseudolig_batch_idces], dim=0)
                    curr_subbatch_idces = torch.cat([curr_subbatch_idces, pseudolig_subbatch_idces + curr_subbatch_idces.max() + 1], dim=0)
                    curr_real_ligand_mask = torch.cat([curr_real_ligand_mask, pseudoligand_node_mask], dim=0)
                else:
                    curr_lig_coords, curr_lig_nodes, curr_lig_atomic_numbers, curr_batch_idces, curr_subbatch_idces = pseudolig_coords, pseudolig_nodes, pseudolig_atomic_numbers, pseudolig_batch_idces, pseudolig_subbatch_idces
                    curr_real_ligand_mask = pseudoligand_node_mask
            
            # Use available ligand data to construct the graphs.
            if curr_lig_coords is not None and curr_lig_nodes is not None and curr_batch_idces is not None and curr_subbatch_idces is not None and curr_lig_atomic_numbers is not None and curr_real_ligand_mask is not None:
                curr_complex_sequence_indices = self.sequence_indices[curr_batch_prot_mask]
                curr_chi_angles = self.chi_angles[curr_batch_prot_mask]
                curr_phi_angles = self.phi_psi_angles[curr_batch_prot_mask][:, 0]

                # Remove pseudoligand-adjacent residues from the edges.
                curr_complex_backbone_coords = curr_complex_backbone_coords[~residues_to_drop_mask]
                curr_complex_sequence_indices = curr_complex_sequence_indices[~residues_to_drop_mask]
                curr_chi_angles = curr_chi_angles[~residues_to_drop_mask]
                curr_phi_angles = curr_phi_angles[~residues_to_drop_mask]

                # Compute the ligand-protein and protein_protein graphs.
                curr_prot_prot_eidx = knn_graph(curr_complex_backbone_coords[:, 1], k=pr_pr_knn_graph_k, loop=True)
                curr_lig_lig_eidx = knn_graph(curr_lig_coords, k=lig_lig_knn_graph_k, loop=True, batch=curr_subbatch_idces) 
                curr_lig_prot_eidx = compute_ligand_protein_knn_graph(curr_lig_coords, curr_complex_backbone_coords[:, 1], lig_pr_knn_graph_k, lig_pr_distance_cutoff, 0, 0, curr_lig_lig_eidx, curr_lig_atomic_numbers, use_aliphatic_ligand_hydrogens)

                # Construct full-atom representation of the protein.
                curr_fa_coords = rotamer_builder.build_rotamers(curr_complex_backbone_coords, curr_chi_angles, curr_complex_sequence_indices)
                assert isinstance(curr_fa_coords, torch.Tensor), 'unreachable.'
                curr_fa_coords = curr_fa_coords.float()

                # Compute the first shell mask for the protein nodes.
                first_shell_node_idces = compute_first_shell_node_idces(curr_fa_coords, curr_complex_sequence_indices, curr_lig_coords, curr_lig_atomic_numbers, curr_lig_prot_eidx, 0)
                hbond_connected_idces = torch.empty((0,), dtype=torch.long, device=self.device)

                # Drop out non-subgraph residues with a small probability.
                prot_index_offset_adjustment = curr_complex_backbone_coords.shape[0]
                if torch.rand(1).item() < subgraph_only_dropout_rate and hbond_connected_idces.shape[0] > 0:
                    num_residues = curr_complex_backbone_coords.shape[0]
                    node_eidces_to_keep, first_shell_node_idces, curr_lig_prot_eidx, curr_prot_prot_eidx = drop_non_subgraph_residues(hbond_connected_idces, curr_lig_prot_eidx, curr_prot_prot_eidx, num_residues, 0)

                    # Update the residues to drop mask to reflect the dropped residues while accounting for the fact that curr_lig_pr_edge_index was created without the dropped residues.
                    keep_mask = torch.zeros(num_residues, device=self.device, dtype=torch.bool)
                    keep_mask[node_eidces_to_keep] = True
                    residues_to_drop_mask_ = residues_to_drop_mask.clone()
                    residues_to_drop_mask_[~residues_to_drop_mask] = ~keep_mask
                    residues_to_drop_mask = residues_to_drop_mask_

                    # Track offsets between complexes to ensure final edge_index is correct.
                    prot_index_offset_adjustment = node_eidces_to_keep.shape[0]

                # Compute the ligand-ligand graph.
                curr_lig_lig_eidx += lig_index_offset
                curr_lig_prot_eidx[0, :] += lig_index_offset
                curr_lig_prot_eidx[1, :] += prot_index_offset
                curr_prot_prot_eidx += prot_index_offset

                # Update the first shell mask to reflect the dropped residues.
                curr_first_shell_mask = curr_first_shell_mask[~residues_to_drop_mask]
                curr_first_shell_mask[first_shell_node_idces] = True

                # Update the offsets for the next complex.
                lig_index_offset += curr_lig_coords.shape[0]
                prot_index_offset += prot_index_offset_adjustment

                # Track the generated tensors:
                all_lig_coords.append(curr_lig_coords)
                all_lig_nodes.append(curr_lig_nodes)
                all_lig_lig_eidces.append(curr_lig_lig_eidx)
                all_lig_prot_eidces.append(curr_lig_prot_eidx)
                all_prot_prot_eidces.append(curr_prot_prot_eidx)
                all_lig_batch_indices.append(curr_batch_idces)
                all_lig_subbatch_indices.append(curr_subbatch_idces)
                all_lig_node_masks.append(curr_real_ligand_mask)
                all_lig_atomic_numbers.append(curr_lig_atomic_numbers)

            # Update the full batch node mask to reflect the dropped residues.
            first_shell_masks.append(curr_first_shell_mask)
            full_batch_node_mask[curr_batch_prot_mask] = ~residues_to_drop_mask

        # Merge all the ligand nodes and masks into a single tensor.
        all_lig_coords = torch.cat(all_lig_coords, dim=0) if len(all_lig_coords) > 0 else torch.empty((0, 3), dtype=torch.float, device=self.device)
        all_lig_nodes = torch.cat(all_lig_nodes, dim=0) if len(all_lig_nodes) > 0 else torch.empty((0, 26), dtype=torch.float, device=self.device)
        all_lig_batch_indices = torch.cat(all_lig_batch_indices, dim=0) if len(all_lig_batch_indices) > 0 else torch.empty((0,), dtype=torch.long, device=self.device)
        all_lig_subbatch_indices = torch.cat(all_lig_subbatch_indices, dim=0) if len(all_lig_subbatch_indices) > 0 else torch.empty((0,), dtype=torch.long, device=self.device)
        all_prot_prot_eidces = torch.cat(all_prot_prot_eidces, dim=-1) if len(all_prot_prot_eidces) > 0 else torch.empty((2, 0), dtype=torch.long, device=self.device)
        all_lig_prot_eidces = torch.cat(all_lig_prot_eidces, dim=-1) if len(all_lig_prot_eidces) > 0 else torch.empty((2, 0), dtype=torch.long, device=self.device)
        all_lig_lig_eidces = torch.cat(all_lig_lig_eidces, dim=-1) if len(all_lig_lig_eidces) > 0 else torch.empty((2, 0), dtype=torch.long, device=self.device)
        all_lig_node_masks = torch.cat(all_lig_node_masks, dim=0) if len(all_lig_node_masks) > 0 else torch.empty((0,), dtype=torch.bool, device=self.device)
        all_lig_atomic_numbers = torch.cat(all_lig_atomic_numbers, dim=0) if len(all_lig_atomic_numbers) > 0 else torch.empty((0,), dtype=torch.long, device=self.device)

        # Mask out dropped residue metadata and update the first shell mask.
        self._apply_mask_to_residue_metadata_tensors(full_batch_node_mask)
        first_shell_masks = torch.cat(first_shell_masks, dim=0) if len(first_shell_masks) > 0 else torch.empty((0,), dtype=torch.bool, device=self.device)
        self.first_shell_ligand_contact_mask = first_shell_masks

        # After dropping out sample-adjacent backbone coordinates, compute the edge_distance_information.
        noised_lig_coords = all_lig_coords + (ligand_training_noise * torch.randn_like(all_lig_coords))

        # Compute edge distances for protein_protein, ligand-protein, and ligand-ligand edges.
        self.pr_pr_edge_index = all_prot_prot_eidces
        self.lig_pr_edge_index = all_lig_prot_eidces
        self.lig_pr_edge_distance = torch.cdist(self.backbone_coords[all_lig_prot_eidces[1]].double(), noised_lig_coords[all_lig_prot_eidces[0]].unsqueeze(1).double()).squeeze(-1).float()
        self.pr_pr_edge_distance = torch.cdist(self.backbone_coords[all_prot_prot_eidces[0]], self.backbone_coords[all_prot_prot_eidces[1]]).flatten(start_dim=1)

        assert all_lig_nodes.shape[0] == all_lig_node_masks.shape[0], f'test: {all_lig_nodes.shape, all_lig_node_masks.shape}.'

        lig_lig_edge_distance = torch.cdist(noised_lig_coords[all_lig_lig_eidces[0]].unsqueeze(1).double(), noised_lig_coords[all_lig_lig_eidces[1]].unsqueeze(1).double()).flatten().float()
        self.ligand_data = LigandData(
            all_lig_nodes, noised_lig_coords, all_lig_batch_indices, 
            all_lig_subbatch_indices, all_lig_lig_eidces, lig_lig_edge_distance, 
            all_lig_node_masks, all_lig_atomic_numbers
        )
    
    def generate_decoding_order(self, stack_tensors: bool = False) -> None:
        """
        Inputs:
            stack_tensors (bool): 
                If False, decoding order is a (N, ) tensor of indices into the (N, ) dimensional tensors.
                If True, stacks decoding orders into a batch dimension for parallel decoding.
                    Used for autoregressive sampling in model.sample function.
                    Results in (B, sub_N_max) tensor of indices into the (N, ) dimensional tensors.
        """
        if not stack_tensors:
            self.decoding_order = _masked_sort_for_decoding_order(self.chain_mask, self.extra_atom_contact_mask)
        else:

            batched_decoding_orders = []
            for batch_idx in range(self.batch_indices.max() + 1):
                curr_batch_mask = self.batch_indices == batch_idx
                output = _masked_sort_for_decoding_order(self.chain_mask, self.extra_atom_contact_mask, curr_batch_mask)
                batched_decoding_orders.append(output)

            output = []
            max_size = max([x.shape[0] for x in batched_decoding_orders])
            for tensor in batched_decoding_orders:
                output.append(torch.cat([tensor, torch.full((max_size - tensor.shape[0],), torch.nan, device=self.device)]))
            self.decoding_order = torch.stack(output)

    def _apply_mask_to_residue_metadata_tensors(self, mask: torch.Tensor) -> None:
        """
        Automatically applies mask of mask.shape[0] to any tensors of the same shape.
        """
        list_of_residue_metadata_tensor_keys = [
            'sequence_indices', 'chi_angles', 'backbone_coords', 'residue_burial_counts', 
            'batch_indices', 'chain_indices', 'resnum_indices', 'chain_mask', 'sampled_chain_mask', 'extra_atom_contact_mask', 
            'msa_data', 'msa_depth_weight', 'phi_psi_angles', 'sidechain_contact_number', 
            'first_shell_ligand_contact_mask', 'sc_mediated_hbond_counts'
        ]
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor) and k in list_of_residue_metadata_tensor_keys:
                self.__dict__[k] = v[mask]

        assert self.decoding_order is None, "Decoding order is set, implement filtering out masked residues in order or call generate_decoding_order after masking."

    def write_pdb_file(self, batch_idx: int, filename: str, rotamer_builder: RotamerBuilder, ligand_featurizer: LigandFeaturizer, real_ligands_only=True, prot_betas=None, lig_betas=None):
        """
        Writes a batch index in the batch to a PDB file to inspect the protein-(pseudo)ligand(s) complexes being input into the model.
        """
        assert self.ligand_data is not None, "Must compute ligand graphs before batch debugging file."
        assert self.ligand_data.real_ligand_mask is not None

        target_batch_mask = self.batch_indices == batch_idx
        ligand_batch_mask = self.ligand_data.lig_batch_indices == batch_idx

        debug = self.ligand_data.real_ligand_mask[ligand_batch_mask].any().item()
        if (ligand_batch_mask.sum().item() == 0 or not debug) and real_ligands_only:
            return False

        curr_backbone = self.backbone_coords[target_batch_mask]
        curr_seq = self.sequence_indices[target_batch_mask]
        curr_chi = self.chi_angles[target_batch_mask]
        curr_fs_mask = self.first_shell_ligand_contact_mask[target_batch_mask]
        if prot_betas is not None:
            curr_betas = prot_betas[target_batch_mask]

        curr_fa_coords = rotamer_builder.build_rotamers(curr_backbone, curr_chi, curr_seq, add_nonrotatable_hydrogens=True)
        assert isinstance(curr_fa_coords, torch.Tensor), "Rotamer builder didn't return a tensor of coordinates."

        protein = create_prody_protein_from_coordinate_matrix(curr_fa_coords.cpu(), curr_seq.cpu())

        # Set the B-factors of the first shell residues to 2.0.
        for idx, res in enumerate(protein.iterResidues()):
            if prot_betas is None:
                if curr_fs_mask[idx]:
                    res.setBetas([2.0] * len(res))
            else:
                res.setBetas([curr_betas[idx, :len(res)].numpy()]) # type: ignore

        ligand = pr.AtomGroup()
        atoms = ligand_featurizer.encoding_to_atom_string_sequence(self.ligand_data.lig_nodes[ligand_batch_mask])
        all_coords = []
        for out in self.ligand_data.lig_coords[ligand_batch_mask].cpu():
            all_coords.append(out.cpu().numpy())
        
        ligand.setCoords(all_coords)
        ligand.setNames(atoms) # type: ignore
        ligand.setResnames(['LIG'] * len(all_coords)) # type: ignore
        ligand.setResnums(self.ligand_data.lig_subbatch_indices[ligand_batch_mask].cpu().numpy()) # type: ignore
        ligand.setChids(['X'] * len(all_coords)) # type: ignore
        ligand.setOccupancies([1.0] * len(all_coords)) # type: ignore
        if lig_betas is None:
            ligand.setBetas([1.0] * len(all_coords)) # type: ignore
        else:
            ligand.setBetas(lig_betas[ligand_batch_mask]) # type: ignore

        # Write the protein-ligand complex to a PDB file with the name of the protein at the top as a title.
        name = ';'.join([x for x in set([self.pdb_codes[x] for x in self.chain_indices[target_batch_mask].cpu().numpy()])])
        protein.setTitle(name)
        pr.writePDB(filename, protein + ligand)

        return True


def drop_non_subgraph_residues(first_shell_node_idces: torch.Tensor, curr_lig_pr_edge_index: torch.Tensor, curr_pr_pr_edge_index: torch.Tensor, num_protein_residues: int, prot_index_offset: int):
    """
    Identifies residues around ligands to use as subgraphs and identifies non-adjacent residues to drop from the graph.
    """

    offset_adjusted_lig_pr_edge_index = curr_lig_pr_edge_index[1, :] - prot_index_offset
    offset_adjusted_pr_pr_edge_index = curr_pr_pr_edge_index - prot_index_offset

    # Sample between one-half to all from randomly shuffled residues in the first shell.
    # end_index = torch.randint(first_shell_node_idces.shape[0] // 2, first_shell_node_idces.shape[0] + 1, (1,)).item()
    end_index = first_shell_node_idces.shape[0] # Samples all
    fs_shuffle_indices = torch.randperm(first_shell_node_idces.shape[0], device=first_shell_node_idces.device)
    sampled_fs_indices = first_shell_node_idces[fs_shuffle_indices[:end_index]]

    # Select 0 to an equal number of randomly shuffled non-first_shell but ligand adjacent residues to drop.
    # ligand_adjacent_residues = offset_adjusted_lig_pr_edge_index.unique()
    # non_fs_ligand_residues = ligand_adjacent_residues[~torch.isin(ligand_adjacent_residues, first_shell_node_idces)]
    # non_fs_shuffle_indices = torch.randperm(non_fs_ligand_residues.shape[0], device=non_fs_ligand_residues.device)
    # end_index = torch.randint(0, min(non_fs_ligand_residues.shape[0], sampled_fs_indices.shape[0]) + 1, (1,)).item()
    # sampled_not_fs_indices = non_fs_ligand_residues[non_fs_shuffle_indices[:end_index]]

    # get the indices we want to keep, and compute new indices.
    # sorted_indices_to_keep, _ = torch.sort(torch.cat([sampled_fs_indices, sampled_not_fs_indices]))
    sorted_indices_to_keep, _ = torch.sort(sampled_fs_indices)
    new_indices = torch.arange(sorted_indices_to_keep.shape[0], device=first_shell_node_idces.device)

    # Adjust the protein component of protein-ligand and protein-protein indices.
    lig_pr_edge_mask = torch.isin(offset_adjusted_lig_pr_edge_index, sorted_indices_to_keep)
    pr_pr_edge_mask = torch.isin(offset_adjusted_pr_pr_edge_index, sorted_indices_to_keep).all(dim=0)

    # Create a map from the old protein indices to the new protein indices.
    old_to_new_indices = torch.full((num_protein_residues,), -1, dtype=torch.long, device=first_shell_node_idces.device)
    old_to_new_indices[sorted_indices_to_keep] = new_indices

    # Update the protein indices in the ligand-protein edge index for those that are kept.
    new_lig_pr_edges = torch.stack([
        curr_lig_pr_edge_index[0, lig_pr_edge_mask],
        old_to_new_indices[offset_adjusted_lig_pr_edge_index[lig_pr_edge_mask]] + prot_index_offset
    ])
    new_pr_pr_edges = old_to_new_indices[offset_adjusted_pr_pr_edge_index[:, pr_pr_edge_mask]] + prot_index_offset
    new_fs_indices = old_to_new_indices[sampled_fs_indices]

    # Returns the node indices to keep and the new ligand-protein edge index.
    return sorted_indices_to_keep, new_fs_indices, new_lig_pr_edges, new_pr_pr_edges


def compute_first_shell_node_idces(
    fa_coords: torch.Tensor, seq_indices: torch.Tensor, ligand_coords: torch.Tensor, ligand_atomic_number_idces: torch.Tensor,
    curr_lig_pr_edge_index: torch.Tensor, prot_index_offset: int
) -> torch.Tensor:
    """
    Identifies residues within the first shell of a pseudoligand and returns the indices of those residues.
    """

    # Get the protein indices of the ligand-protein edge index.
    protein_eidx = (curr_lig_pr_edge_index[1, :] - prot_index_offset).unique()
    putative_contacting_residues = fa_coords[protein_eidx]
    seq_indices_ = seq_indices[protein_eidx]

    is_gly_mask = (seq_indices_ == aa_short_to_idx['G']) | (seq_indices_ == aa_short_to_idx['X'])
    hydrogen_mask = (ligand_atomic_number_idces == 0)

    # Compute the distance between the ligand heavy atoms and the putative contacting residue heavy atoms.
    contact_mask = torch.zeros_like(protein_eidx, dtype=torch.bool)
    not_gly_contact_mask = (torch.cdist(ligand_coords, putative_contacting_residues[~is_gly_mask][:, 4:]) < HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD).any(dim=-1)[:, ~hydrogen_mask].any(dim=-1)
    gly_contact_mask = (torch.cdist(ligand_coords, putative_contacting_residues[is_gly_mask][:, 1].unsqueeze(1)) < HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD + 0.3).any(dim=-1)[:, ~hydrogen_mask].any(dim=-1)

    contact_mask[~is_gly_mask] = not_gly_contact_mask
    contact_mask[is_gly_mask] = gly_contact_mask

    return protein_eidx[contact_mask] 


def get_list_of_all_paths(path: str) -> list:
    """
    Recursively get a list of all paths of pytorch files in a directory.
    """
    all_paths = []
    for subdir_or_files in os.listdir(path):
        path_to_subdir_or_file = os.path.join(path, subdir_or_files)
        if path_to_subdir_or_file.endswith('.pt'):
            all_paths.append(path_to_subdir_or_file)
        elif os.path.isdir(path_to_subdir_or_file):
            all_paths.extend(get_list_of_all_paths(path_to_subdir_or_file))
    return all_paths


def compute_ligand_protein_knn_graph(
        curr_lig_coords, curr_complex_ca_coords, lig_pr_k, distance_cutoff, 
        lig_index_offset, prot_index_offset, lig_lig_eidx, lig_atomic_number_idces, use_aliphatic_ligand_hydrogens
) -> torch.Tensor:
    # Identify the residues with CA coordinates within connection radius to ligand.
    ca_distances = torch.cdist(curr_complex_ca_coords.double(), curr_lig_coords.unsqueeze(0).double()).squeeze(0)

    num_ligand_atoms = curr_lig_coords.shape[0]
    if not use_aliphatic_ligand_hydrogens:
        # Use the ligand-ligand graph to identify the ligand atoms that are hydrogens covalently bonded to a carbon with a mask for curr_lig_coords.
        lig_lig_eidx_ = lig_lig_eidx[:, lig_lig_eidx[0] != lig_lig_eidx[1]]
        is_within_covalent_hydrogen_distance = (torch.cdist(curr_lig_coords[lig_lig_eidx_[0]].unsqueeze(1), curr_lig_coords[lig_lig_eidx_[1]].unsqueeze(1)) < COVALENT_HYDROGEN_BOND_MAX_DISTANCE).flatten().float()
        sink_is_within_covalent_hydrogen_distance = scatter(is_within_covalent_hydrogen_distance, lig_lig_eidx_[1], reduce='max', dim_size=curr_lig_coords.shape[0]).bool()
        is_aliphatic_hydrogen_mask = (lig_atomic_number_idces == 0) & sink_is_within_covalent_hydrogen_distance

        # Set the distances to the aliphatic hydrogens to infinity.
        ca_distances[:, is_aliphatic_hydrogen_mask] = torch.finfo(ca_distances.dtype).max
        num_ligand_atoms = (~is_aliphatic_hydrogen_mask).sum().item()

    # Identify the protein atoms within the connection radius to the ligand.
    connected_bb_indices = (ca_distances < distance_cutoff).sum(dim=1).nonzero().flatten()

    # Update KNN-k parameter to handle case where we have fewer than k ligand atoms.
    curr_lig_pr_k = min(lig_pr_k, num_ligand_atoms)

    # Construct a KNN graph between the connected protein atoms and the ligand atoms.
    #   Ordered as lig -> prot.
    curr_lig_pr_edge_index = torch.stack([
        ca_distances[connected_bb_indices].argsort(dim=1)[:, :curr_lig_pr_k].flatten() + lig_index_offset, # Keep indices for K nearest cg atoms by sorting along column dimension.
        connected_bb_indices.unsqueeze(-1).expand(-1, curr_lig_pr_k).flatten() + prot_index_offset # Duplicates backbone indices K times.
    ])
    return curr_lig_pr_edge_index


def invert_dict(d: dict) -> dict:
    clusters = defaultdict(list)
    for k, v in d.items():
        clusters[v].append(k)
    return dict(clusters)


def chain_list_to_protein_chain_dict(chain_list: list) -> dict:
    """
    Takes a list of bioassemblies+segment+chains and returns a dictionary 
    mapping pdb code to a list of assemblies and chains in a given sequence cluster.
    """

    bioasmb_list = defaultdict(list)
    for chain in chain_list:
        pdb_code, asmb_chain_id = chain.split('_')
        bioasmb_list[pdb_code].append(asmb_chain_id)

    return dict(bioasmb_list)


def get_complex_len(complex_data: dict) -> int:
    """
    Chains are indicated with segment/chain tuples so only these have size attr.
    """
    
    # Create a sorta-resnum by adding every 10 ligand atoms to the resnum count.
    num_lig_coords = 0
    if 'ligands' in complex_data:
        num_lig_coords += sum(len(x) for x in complex_data['ligands']['coords'])
    if 'xtal_additives' in complex_data:
        num_lig_coords += sum(len(x) for x in complex_data['xtal_additives']['coords'])

    num_protein_nodes = sum([x['size'] for y, x in complex_data.items() if isinstance(y, tuple)])

    return num_protein_nodes + num_lig_coords


def pdb_and_chain_to_code(pdb_code: str, chain_tup: Tuple[str, str]) -> str:
    return '-'.join([pdb_code, *chain_tup])


def compute_msa_depth_weight(msa_depth: float, median: float = 96.0) -> float:
    """
    Using the median MSA depth across the dataset (computed during parsing), computes a weight 
    for each MSA depth to avoid over penalizing shallow MSAs.
    """
    return min(msa_depth / (2 * median), 1.0)


@torch.no_grad()
def idealize_backbone_coords(bb_coords, phi_psi_angles) -> torch.Tensor:
    # Expand copies of the idealized frames to match the number of frames in the batch
    ideal_ala_coords = ideal_prot_aa_coords[aa_short_to_idx['A']]
    ideal_frames_exp = ideal_ala_coords[[hydrogen_extended_dataset_atom_order['A'].index(x) for x in ('N', 'CA', 'C')]].to(bb_coords.device).unsqueeze(0).expand(bb_coords.shape[0], -1, -1)

    # Align the idealized frames from the origin to the actual backbone frames
    frame_alignment_matrices = compute_alignment_matrices(bb_coords[:, [0, 1, 3]], ideal_frames_exp)
    N, CA, C = apply_transformation(ideal_frames_exp, *frame_alignment_matrices).unbind(dim=1)

    # Compute virtual CB coordinates for idealized frames.
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    # Compute the oxygen coordinates for the ideal frames (psi-dependent so need to do some math).
    dihedral_angles_rad = (phi_psi_angles[:, 1].unsqueeze(1) + 180.0).nan_to_num().deg2rad() # just use 0.0 deg for missing angles.
    bond_angles = torch.full_like(dihedral_angles_rad, fill_value=120.8).deg2rad() # Ca-C-O angle in radians
    bond_lengths = torch.full_like(dihedral_angles_rad, fill_value=1.23) # C-O bond length
    O = extend_coordinates(torch.stack([N, CA, C], dim=1), bond_lengths, bond_angles, dihedral_angles_rad)

    return torch.stack([N, CA, CB, C, O], dim=1)


def collate_sampler_data(data: list, use_xtal_additive_ligands: bool, disable_convex_hull: bool = False, recompute_all_cb_atoms: bool = True, disable_ligand_information: bool = False) -> BatchData:
    """
    Given a list of data from protein dataset, creates a BatchData object holding a single batch for input to model.
    Handles converting protein complex data into torch tensors with desired shapes and properties.
    """
    chain_idx = 0
    all_ligand_data = None
    all_batch_data = defaultdict(list)
    for batch_idx, (complex_data, chain_key) in enumerate(data):

        subbatch_idx = 0
        if 'ligands' in complex_data and not disable_ligand_information:
            for (lig_coords, lig_elements) in zip(complex_data['ligands']['coords'], complex_data['ligands']['elements']):
                lig_subbatch_idx = torch.full((lig_coords.shape[0],), subbatch_idx, dtype=torch.long)
                lig_batch_index = torch.full((lig_coords.shape[0],), batch_idx, dtype=torch.long)

                if 'X' in lig_elements:
                    continue

                lig_atomic_numbers = torch.tensor([atom_to_atomic_number[x] for x in lig_elements])
                assert lig_coords.shape[0] == lig_atomic_numbers.shape[0], "Coords and element tensor shapes must match!"

                lig_data = UnprocessedLigandData(lig_atomic_numbers, lig_coords.float(), lig_batch_index, lig_subbatch_idx)
                if all_ligand_data is None:
                    all_ligand_data = lig_data
                else:
                    all_ligand_data = all_ligand_data.extend(lig_data)
                subbatch_idx += 1
        
        if 'xtal_additives' in complex_data and use_xtal_additive_ligands and not disable_ligand_information:
            for (lig_coords, lig_elements) in zip(complex_data['xtal_additives']['coords'], complex_data['xtal_additives']['elements']):
                lig_subbatch_idx = torch.full((lig_coords.shape[0],), subbatch_idx, dtype=torch.long)
                lig_batch_index = torch.full((lig_coords.shape[0],), batch_idx, dtype=torch.long)
                if 'X' in lig_elements:
                    continue
                lig_atomic_numbers = torch.tensor([atom_to_atomic_number[x] for x in lig_elements])
                assert lig_coords.shape[0] == lig_atomic_numbers.shape[0], "Coords and element tensor shapes must match!"

                lig_data = UnprocessedLigandData(lig_atomic_numbers, lig_coords.float(), lig_batch_index, lig_subbatch_idx)
                if all_ligand_data is None:
                    all_ligand_data = lig_data
                else:
                    all_ligand_data = all_ligand_data.extend(lig_data)
                subbatch_idx += 1
        
        # Single-chain complexes don't have TM-align data.
        tm_align_map = None
        if chain_key in complex_data['tm_align_scores']:
            tm_align_map = complex_data['tm_align_scores'][chain_key]
        
        curr_pdb_code = complex_data['pdb_code']

        # Loop over the remaining chains and add them to the batch.
        all_backbone_coords = []
        for curr_chain_tup, chain_data in complex_data.items():

            # Chains are represented with key tuples of (segment, chain), assume everything else is metadata.
            if not isinstance(curr_chain_tup, tuple):
                continue

            is_sampled_chain = tuple(chain_key.split('-')[-2:]) == curr_chain_tup

            # Convert chain tuple to chain key by appending pdb code and concatenating with '-'.
            curr_chain_key = pdb_and_chain_to_code(curr_pdb_code, curr_chain_tup)

            # Extract MSA data, pad with zeros if missing.
            msa_data = chain_data['msa_data']
            if chain_data['msa_data'] is None:
                msa_data = torch.zeros(chain_data['size'], 21, dtype=torch.float)

            # Weight MSA data by MSA depth resulting in a float from 0 to 1.
            msa_depth = compute_msa_depth_weight(chain_data['msa_depth'])
            msa_depth = torch.full((chain_data['size'],), msa_depth, dtype=torch.long)

            # If missing TM-align data, set to 1.0 so we mask it by default.
            sampled_chain_to_curr_chain_tm_align_score = 1.0
            if (not tm_align_map is None) and (curr_chain_key in tm_align_map):
                sampled_chain_to_curr_chain_tm_align_score = tm_align_map[curr_chain_key]

            # Provides rotamers for anything not structurally similar, or smaller in size than MAX_PEPTIDE_LENGTH.
            # Recently, anything smaller than MAX_PEPTIDE_LENGTH should have been converted into a ligand.
            if sampled_chain_to_curr_chain_tm_align_score < MIN_TM_SCORE_FOR_SIMILARITY or chain_data['size'] <= MAX_PEPTIDE_LENGTH:
                chain_mask = torch.ones(chain_data['size'], dtype=torch.bool)
            else:
                chain_mask = torch.zeros(chain_data['size'], dtype=torch.bool)

            # Precomputed in ProteinAssemblyDataset.compute_masks
            first_shell_ligand_contact_mask = chain_data['first_shell_ligand_contact_mask']
            extra_atom_contact_mask = chain_data['extra_atom_contact_mask']
            sc_mediated_hbond_counts = chain_data['sc_mediated_hbond_counts']

            # Extract the rest of the chain data.
            sequence_indices = chain_data['sequence_indices'].long()
            resnum_indices = chain_data['seqres_resnums']
            if resnum_indices is None:
                resnum_indices = torch.full((chain_data['size'],), -1, dtype=torch.long)

            if recompute_all_cb_atoms:
                chi_angles = chain_data['virtual_cb_adj_chi_angles']
            else:
                chi_angles = chain_data['chi_angles']

            phi_psi_angles = chain_data['phi_psi_angles']

            # Compute a mask for chi_angles that are missing, second chi angle for Cys to be absent.
            non_nan_chi_mask = (~chi_angles.isnan())
            cys_mask = (sequence_indices == aa_short_to_idx['C'])
            expected_num_chi_mask = (non_nan_chi_mask.sum(dim=-1) == aa_to_chi_angle_mask[sequence_indices].sum(dim=-1))
            expected_num_chi_mask[cys_mask] = non_nan_chi_mask[cys_mask, 0]

            # Set malformed residues to X, Remove the chi angles for residues that are missing them.
            sequence_indices[~expected_num_chi_mask] = aa_short_to_idx['X']
            chi_angles[~expected_num_chi_mask, :] = torch.nan

            contact_number = chain_data['contact_number'].long()
            residue_burial_counts = chain_data['residue_cb_counts']
            backbone_coords = chain_data['backbone_coords'].float()
            batch_indices = torch.full((backbone_coords.shape[0],), batch_idx, dtype=torch.long)
            chain_indices = torch.full((backbone_coords.shape[0],), chain_idx, dtype=torch.long)
            sampled_chain_mask = (torch.zeros_like(chain_mask, dtype=torch.long) + int(is_sampled_chain)).bool()

            all_batch_data['pdb_codes'].append(curr_chain_key)
            all_batch_data['chain_mask'].append(chain_mask)
            all_batch_data['sampled_chain_mask'].append(sampled_chain_mask)
            all_batch_data['extra_atom_contact_mask'].append(extra_atom_contact_mask)
            all_batch_data['first_shell_ligand_contact_mask'].append(first_shell_ligand_contact_mask)
            all_batch_data['sequence_indices'].append(sequence_indices)
            all_batch_data['chi_angles'].append(chi_angles)
            # all_batch_data['backbone_coords'].append(backbone_coords)
            all_batch_data['phi_psi_angles'].append(phi_psi_angles)
            all_batch_data['sidechain_contact_number'].append(contact_number)
            all_batch_data['residue_burial_counts'].append(residue_burial_counts)
            all_batch_data['batch_indices'].append(batch_indices)
            all_batch_data['chain_indices'].append(chain_indices)
            all_batch_data['msa_data'].append(msa_data.float())
            all_batch_data['msa_depth_weight'].append(msa_depth)
            all_batch_data['resnum_indices'].append(resnum_indices)
            all_batch_data['sc_mediated_hbond_counts'].append(sc_mediated_hbond_counts)
            all_backbone_coords.append(backbone_coords)

            chain_idx += 1

        all_backbone_coords = torch.cat(all_backbone_coords, dim=0)
        all_batch_data['backbone_coords'].append(all_backbone_coords)
        
    # Concatenate all the data in the batch dimension.
    outputs = {}
    for i,j in all_batch_data.items():
        if i in ['pdb_codes']:
            outputs[i] = j
        else:
            if None in j:
                assert False, f"Missing data for {i}"
            outputs[i] = torch.cat(j, dim=0)
    
    outputs['unprocessed_ligand_input_data'] = all_ligand_data
    
    # Create a BatchData object and compute the KNN graph.
    output_batch_data = BatchData(**outputs)

    return output_batch_data


def filter_subclusters(subclusters_info: dict, curr_clusters: dict) -> dict:
    """
    Filters out subclusters that are not in the current cluster set.
    """

    filtered_subclusters = defaultdict(lambda: defaultdict(list))
    for cluster, subcluster_dict in subclusters_info.items():
        if cluster in curr_clusters:
            for subcluster_centroid, subcluster_list in subcluster_dict.items():
                for subcluster_entry in subcluster_list:
                    if subcluster_entry in curr_clusters[cluster]:
                        filtered_subclusters[cluster][subcluster_centroid].append(subcluster_entry)
    
    return filtered_subclusters
    

class ProteinAssemblyDataset(Dataset):
    """
    Dataset where every biological assembly is a separate index.
    """
    def __init__(self, params):

        shelve_path = params['raw_dataset_path']
        assert os.path.exists(shelve_path + '.dat') , f"Path to shelve file does not exist: {shelve_path}"
        self.data = shelve.open(shelve_path, 'r', protocol=5)
        self.index_to_key = list(self.data.keys())
    
    def __del__(self) -> None:
        self.data.close()

    def __len__(self) -> int:
        return len(self.data.keys())

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        pdb_code = self.index_to_key[index]
        return self.data[pdb_code], pdb_code
    
    def compute_masks(self, rotamer_builder: RotamerBuilder, params: dict, device: torch.device = torch.device('cpu')) -> None:
        """
        Want to pre-compute the contact masks for all the extra atom coords (to mask residues out from the loss),
        as well as the first shell masks for the various ligands.

        Run on dataset with:
            from utils.pretraining_dataset import ProteinAssemblyDataset
            protein_dataset = ProteinAssemblyDataset(params)
            params = {
                'dataset_path': '/scratch/bfry/torch_bioasmb_dataset_complete',
                'output_dataset_path': '/scratch/bfry/torch_bioasmb_dataset_complete_masked/',
             }
            def construct_output_path(output_path, input_path):

                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                else:
                    print(f"Warning, output directory path exists. {output_path}")

                for subdir in os.listdir(input_path):
                    subdir_path = os.path.join(output_path, subdir)
                    if not os.path.exists(subdir_path):
                        os.mkdir(subdir_path)
            
            construct_output_path(params['output_dataset_path'], params['dataset_path'])
            protein_dataset.compute_masks(model.rotamer_builder, params, device)
        """

        for idx in tqdm(range(len(self)), total=len(self), desc='Computing masks.', dynamic_ncols=True):
            output_data, pdb_code = self[idx]
            all_ligand_data = None
            all_extra_coords_data = None

            output_path = os.path.join(params['output_dataset_path'], pdb_code[1:3], pdb_code)
            if os.path.exists(output_path):
                continue

            ligands_to_move_to_extra_atoms = []
            subbatch_idx = 0
            if 'ligands' in output_data:
                for (lig_coords, lig_elements) in zip(output_data['ligands']['coords'], output_data['ligands']['elements']):
                    # lig_batch_index = torch.full((lig_coords.shape[0],), batch_idx, dtype=torch.long)
                    lig_subbatch_idx = torch.full((lig_coords.shape[0],), subbatch_idx, dtype=torch.long, device=device)
                    lig_batch_index = torch.zeros(lig_coords.shape[0], device=device)

                    if 'X' in lig_elements:
                        ligands_to_move_to_extra_atoms.append((lig_coords, lig_elements))
                        continue

                    lig_atomic_numbers = torch.tensor([atom_to_atomic_number[x] for x in lig_elements], device=device)
                    assert lig_coords.shape[0] == lig_atomic_numbers.shape[0], "Coords and element tensor shapes must match!"

                    lig_data = UnprocessedLigandData(lig_atomic_numbers, lig_coords.to(device), lig_batch_index, lig_subbatch_idx)
                    if all_ligand_data is None:
                        all_ligand_data = lig_data
                    else:
                        all_ligand_data.extend(lig_data)

                    subbatch_idx += 1

            if 'xtal_additives' in output_data:
                for (lig_coords, lig_elements) in zip(output_data['xtal_additives']['coords'], output_data['xtal_additives']['elements']):
                    # lig_batch_index = torch.full((lig_coords.shape[0],), batch_idx, dtype=torch.long)
                    lig_subbatch_idx = torch.full((lig_coords.shape[0],), subbatch_idx, dtype=torch.long, device=device)
                    lig_batch_index = torch.zeros(lig_coords.shape[0], device=device)
                    if 'X' in lig_elements:
                        ligands_to_move_to_extra_atoms.append((lig_coords, lig_elements))
                        continue
                    lig_atomic_numbers = torch.tensor([atom_to_atomic_number[x] for x in lig_elements], device=device)
                    assert lig_coords.shape[0] == lig_atomic_numbers.shape[0], "Coords and element tensor shapes must match!"

                    lig_data = UnprocessedLigandData(lig_atomic_numbers, lig_coords.to(device), lig_batch_index, lig_subbatch_idx)
                    if all_ligand_data is None:
                        all_ligand_data = lig_data
                    else:
                        all_ligand_data.extend(lig_data)

                    subbatch_idx += 1

            if 'extra_atoms' in output_data:
                subbatch_idx = 0
                for (lig_coords, lig_elements) in zip(output_data['extra_atoms']['coords'], output_data['extra_atoms']['elements']):
                    # lig_batch_index = torch.full((lig_coords.shape[0],), batch_idx, dtype=torch.long)
                    lig_subbatch_idx = torch.full((lig_coords.shape[0],), subbatch_idx, dtype=torch.long, device=device)
                    lig_batch_index = torch.zeros(lig_coords.shape[0], device=device)
                    lig_atomic_numbers = torch.tensor([atom_to_atomic_number[x] if x in atom_to_atomic_number else -1 for x in lig_elements], device=device)
                    assert lig_coords.shape[0] == lig_atomic_numbers.shape[0], "Coords and element tensor shapes must match!"

                    lig_data = UnprocessedLigandData(lig_atomic_numbers, lig_coords.to(device), lig_batch_index, lig_subbatch_idx)
                    if all_extra_coords_data is None:
                        all_extra_coords_data = lig_data
                    else:
                        all_extra_coords_data.extend(lig_data)
                    subbatch_idx += 1

                for (lig_coords, lig_elements) in ligands_to_move_to_extra_atoms:
                    lig_subbatch_idx = torch.full((lig_coords.shape[0],), subbatch_idx, dtype=torch.long, device=device)
                    lig_batch_index = torch.zeros(lig_coords.shape[0], device=device)
                    lig_atomic_numbers = torch.tensor([atom_to_atomic_number[x] if x in atom_to_atomic_number else -1 for x in lig_elements], device=device)
                    assert lig_coords.shape[0] == lig_atomic_numbers.shape[0], "Coords and element tensor shapes must match!"

                    lig_data = UnprocessedLigandData(lig_atomic_numbers, lig_coords.to(device), lig_batch_index, lig_subbatch_idx)
                    if all_extra_coords_data is None:
                        all_extra_coords_data = lig_data
                    else:
                        all_extra_coords_data.extend(lig_data)
                    subbatch_idx += 1

            output_data_updates = {}
            for key, val in output_data.items():

                if isinstance(key, tuple):
                    backbone_coords = val['backbone_coords'].to(device)
                    sequence_indices = val['sequence_indices'].to(device).long()
                    chi_angles = val['chi_angles'].to(device)
                    structure_coords = rotamer_builder.build_rotamers(backbone_coords, chi_angles, sequence_indices, add_nonrotatable_hydrogens=False)
                    assert isinstance(structure_coords, torch.Tensor), f"Expected torch.Tensor, got {type(structure_coords)}." # Appease type checker.

                    if all_ligand_data is not None:
                        is_hydrogen_mask = all_ligand_data.lig_atomic_numbers == 1
                        ligand_heavy_atoms = all_ligand_data.lig_coords[~is_hydrogen_mask]
                        # (N x 15 x 3) @ (1 x M x 3) -> (N x 15 x M)
                        first_shell_contact_mask = torch.cdist(structure_coords.double(), ligand_heavy_atoms.double().unsqueeze(0)) < HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD
                        first_shell_contact_mask = first_shell_contact_mask.any(dim=-1).any(dim=-1)
                    else:
                        first_shell_contact_mask = torch.zeros((backbone_coords.shape[0],), dtype=torch.bool)

                    if all_extra_coords_data is not None:
                        is_hydrogen_mask = all_extra_coords_data.lig_atomic_numbers == 1
                        extra_heavy_atoms = all_extra_coords_data.lig_coords[~is_hydrogen_mask]
                        extra_atom_contact_mask = torch.cdist(structure_coords.double(), extra_heavy_atoms.double().unsqueeze(0)) < HEAVY_ATOM_CONTACT_DISTANCE_THRESHOLD
                        extra_atom_contact_mask = extra_atom_contact_mask.any(dim=-1).any(dim=-1)
                    else:
                        extra_atom_contact_mask = torch.zeros((backbone_coords.shape[0],), dtype=torch.bool)

                    val['first_shell_ligand_contact_mask'] = first_shell_contact_mask.cpu()
                    val['extra_atom_contact_mask'] = extra_atom_contact_mask.cpu()

                output_data_updates[key] = val
            
            torch.save(output_data_updates, output_path)

    def compute_hydrogen_bond_contact_number(self, hbond_network_detector: RigorousHydrogenBondNetworkDetector, rotamer_builder: RotamerBuilder, ligand_featurizer: LigandFeaturizer, params: dict, device: torch.device) -> None:
        with shelve.open(params['output_dataset_path'], 'c') as new_db:
            for idx in tqdm(range(len(self)), total=len(self), desc='Computing hydrogen bond contact number.', dynamic_ncols=True):
                output_data, pdb_code = self[idx]

                # Build the the protein.
                all_full_atom_coords = None
                all_polymer_sequence_idces = None
                all_sequence_indices = None
                all_chain_idces = None
                all_phi_angles = None
                chain_idx = -1
                chain_key_to_index = {}
                for key, val in output_data.items():

                    if isinstance(key, tuple):
                        chain_idx += 1
                        chain_key_to_index[key] = chain_idx

                        backbone_coords = val['backbone_coords'].to(device)
                        sequence_indices = val['sequence_indices'].to(device).long()
                        chi_angles = val['chi_angles'].to(device)
                        phi_angles = val['phi_psi_angles'][:, 0].to(device)

                        structure_coords = rotamer_builder.build_rotamers(backbone_coords, chi_angles, sequence_indices, add_nonrotatable_hydrogens=True)
                        assert isinstance(structure_coords, torch.Tensor), f"Expected torch.Tensor, got {type(structure_coords)}."
                        seqres_resnums = val['seqres_resnums'].to(device)

                        if all_full_atom_coords is not None and all_polymer_sequence_idces is not None and all_chain_idces is not None and all_sequence_indices is not None and all_phi_angles is not None:
                            all_full_atom_coords = torch.cat([all_full_atom_coords, structure_coords], dim=0)
                            all_polymer_sequence_idces = torch.cat([all_polymer_sequence_idces, seqres_resnums], dim=0)
                            all_sequence_indices = torch.cat([all_sequence_indices, sequence_indices], dim=0)
                            all_chain_idces = torch.cat([all_chain_idces, torch.full((structure_coords.shape[0],), chain_idx, dtype=torch.long, device=device)], dim=0)
                            all_phi_angles = torch.cat([all_phi_angles, phi_angles], dim=0)
                        else:
                            all_full_atom_coords = structure_coords
                            all_polymer_sequence_idces = seqres_resnums
                            all_chain_idces = torch.full((structure_coords.shape[0],), chain_idx, dtype=torch.long, device=device)
                            all_sequence_indices = sequence_indices
                            all_phi_angles = phi_angles
                
                # After inspecting the reason why this happens, seems like all chains are < peptide length and therefore there are no protein residues.
                if all_full_atom_coords is None:
                    print(f"Skipping {pdb_code} due to missing full atom coords.")
                    continue

                assert all_full_atom_coords is not None, "No full atom coords found for assembly."
                assert all_polymer_sequence_idces is not None, "No polymer sequence idces found for assembly."
                assert all_chain_idces is not None, "No chain idces found for assembly."
                assert all_sequence_indices is not None, "No sequence indices found for assembly."
                assert all_phi_angles is not None, "No phi angles found for assembly."

                ca_coords = all_full_atom_coords[:, 1, :]
                e_idces = knn_graph(ca_coords, k=48, loop=False)

                # Identify Non-Adjacent Source Edges:
                adjacent_edge_mask = torch.abs(all_polymer_sequence_idces[e_idces[0]] - all_polymer_sequence_idces[e_idces[1]]) <= params['num_adjacent_residues_to_drop']
                same_chain_mask = all_chain_idces[e_idces[0]] == all_chain_idces[e_idces[1]]
                e_idces = e_idces[:, ~(adjacent_edge_mask & same_chain_mask)]

                # Compute the residues with sidechain-mediated hydrogen bonding.
                bb_nh_coords = rotamer_builder.impute_backbone_nh_coords(all_full_atom_coords.float(), all_sequence_indices, all_phi_angles.float().unsqueeze(-1))
                sc_mediated_hbond_counts = hbond_network_detector._compute_hbonding_counts(all_full_atom_coords, bb_nh_coords, all_sequence_indices, e_idces, ligand_featurizer)

                for chain_key, chain_idx in chain_key_to_index.items():
                    chain_mask = all_chain_idces == chain_idx
                    sc_mediated_hbond_counts_chain = sc_mediated_hbond_counts[chain_mask]
                    output_data[chain_key]['sc_mediated_hbond_counts'] = sc_mediated_hbond_counts_chain.cpu()

                new_db[pdb_code] = output_data
    

class UnclusteredProteinChainDataset(Dataset):
    """
    Dataset where every pdb_assembly-segment-chain is a separate index.
    """
    def __init__(self, params):

        metadata_shelve_path = str(params['metadata_dataset_path'])# + ('.debug' if params['debug'] else '')
        if not os.path.exists(metadata_shelve_path + '.dat'):
            print("Computing dataset metadata shelve, this only needs to run once.")
            compute_metadata_from_raw_data_shelve(params['raw_dataset_path'], metadata_shelve_path, params['debug'])

        self.pdb_code_to_complex_data = shelve.open(params['raw_dataset_path'], 'r', protocol=5)

        metadata = shelve.open(metadata_shelve_path, 'r', protocol=5)
        self.chain_key_to_index = metadata['chain_key_to_index']
        self.index_to_complex_size = metadata['index_to_complex_size']
        self.index_to_chain_key = metadata['index_to_chain_key']
        self.index_to_num_ligand_contacting_residues = metadata['index_to_num_ligand_contacting_residues']
        metadata.close()

    def __del__(self) -> None:
        if hasattr(self, 'pdb_code_to_complex_data'):
            self.pdb_code_to_complex_data.close()

    def __len__(self) -> int:
        return len(self.chain_key_to_index)

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        # Take indexes unique to chain and return the complex data for that chain and the chain key.
        chain_key = self.index_to_chain_key[index]
        pdb_code = chain_key.split('-')[0]
        output_data = self.pdb_code_to_complex_data[pdb_code]
        return output_data, chain_key
    
    def write_all_sequence_fasta(self, output_path: str) -> None:
        """
        Writes all sequences longer than MAX_PEPTIDE_LENGTH to a fasta file.

        Run 30% cluster generation with:
            `mmseqs easy-cluster fasta.txt cluster30test tmp30test --min-seq-id 0.3 -c 0.5 --cov-mode 5 --cluster-mode 3`
        """
        output = {}
        # Loop over everything in the dataset.
        for pdb_code, data_dict in tqdm(self.pdb_code_to_complex_data.items(), total=len(self.pdb_code_to_complex_data)):
            for key, sub_data in data_dict.items():
                # Select the chains which are ('Segment', 'Chain') tuples and record crystallized sequence.
                if isinstance(key, tuple):
                    chain_key = "-".join([pdb_code, *key])
                    sequence = sub_data['polymer_seq']
                    output[chain_key] = sequence
        
        # Sort the output by chain_key so the fasta file is sorted.
        output = sorted(output.items(), key=lambda x: x[0])

        # Write the fasta file.
        with open(output_path, 'w') as f:
            for chain_key, sequence in output:
                if sequence is not None and len(sequence) > MAX_PEPTIDE_LENGTH:
                    f.write(f">{chain_key}\n")
                    f.write(f"{sequence}\n")


def compute_metadata_from_raw_data_shelve(path_to_raw_shelve: str, path_to_output_shelve: str, is_debug: bool) -> None:
    """
    Computes metadata from the raw data shelve and creates a new shelve storing that metadata.
    """
    chain_key_to_index = {}
    index_to_complex_size = {}
    index_to_num_ligand_contacting_residues = {}

    idx = 0
    with shelve.open(path_to_raw_shelve, 'r', protocol=5) as db_:
        # Only load the debug chains if we are in debug mode.
        db_keys = list(db_.keys())
        if is_debug:
            db_keys = [x for x in db_keys if x[1:3] == 'w7' or x[:4] == '4jnj']

        # Loop over all the chains and record the chain key and the complex size.
        for pdb_code in tqdm(db_keys, desc='Computing metadata for dataset sampling...', dynamic_ncols=True):
            protein_data = db_[pdb_code]
            protein_complex_len = get_complex_len(protein_data)

            for chain_key, chain_data in protein_data.items():
                if isinstance(chain_key, tuple):
                    chain_key = '-'.join([pdb_code] + list(chain_key))
                    chain_key_to_index[chain_key] = idx
                    index_to_complex_size[idx] = protein_complex_len
                    index_to_num_ligand_contacting_residues[idx] = chain_data['first_shell_ligand_contact_mask'].sum()
                    idx += 1

    # Invert the chain_key_to_index dict to get a cluster to chain mapping.
    index_to_chain_key = {x: y for y,x in chain_key_to_index.items()}

    # Write the metadata to a shelve.
    with shelve.open(path_to_output_shelve, 'c', protocol=5) as db_:
        db_['chain_key_to_index'] = chain_key_to_index
        db_['index_to_complex_size'] = index_to_complex_size
        db_['index_to_chain_key'] = index_to_chain_key
        db_['index_to_num_ligand_contacting_residues'] = index_to_num_ligand_contacting_residues


class ClusteredDatasetSampler(Sampler):
    """
    Samples a single protein complex from precomputed mmseqs clusters.
    Ensures samples drawn evenly by sampling first from sequence clusters, then by pdb_code, then by assembly and chain.
    Iteration returns batched indices for use in UnclusteredProteinChainDataset.
    Pass to a DataLoader as a batch_sampler.
    """
    def __init__(
        self, dataset: UnclusteredProteinChainDataset, params: dict, 
        is_test_dataset_sampler: bool, single_protein_debug: bool = False, 
        seed: Optional[int] = None, subset_pdb_code_list: Optional[List[str]] = None,
    ):
        # Set the random seed for reproducibility and consistent randomness between processes if parallelized.
        if seed is None:
            self.generator = torch.Generator(device='cpu')
        else:
            self.generator = torch.Generator().manual_seed(seed)

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.shuffle = params['sample_randomly']
        self.max_protein_length = params['max_protein_size']
        self.bias_sample_to_proteins_with_ligand_fraction = params['bias_sample_to_proteins_with_ligand_fraction']
        self.min_num_ligand_contacting_residues_for_bias_sample = params['min_num_ligand_contacting_residues_for_bias_sample']

        self.subset_pdb_code_set = set(subset_pdb_code_list) if subset_pdb_code_list is not None else {}

        # Load the cluster data.
        sequence_clusters = pd.read_pickle(params['clustering_dataframe_path'])

        if params['debug']:
            sequence_clusters = sequence_clusters[sequence_clusters.chain.str.find('w7') == 1]

        train_cluster_meta = sequence_clusters[sequence_clusters.is_train & (~sequence_clusters.contaminated) & (~sequence_clusters.strep_structural_chain_contam) & (~sequence_clusters.strep_structural_bioa_contam)]
        test_cluster_meta = sequence_clusters[(~sequence_clusters.is_train) & (~sequence_clusters.contaminated) & (~sequence_clusters.strep_structural_chain_contam) & (~sequence_clusters.strep_structural_bioa_contam)]

        if params['soluble_proteins_only']:
            if len(self.subset_pdb_code_set) > 0:
                raise NotImplementedError("Subset pdb codes not implemented with soluble proteins only.")

            possible_membrane_pdb_codes = set(pd.read_csv(str(Path(__file__).parent.parent / 'files/membrane_excluded_PDBs.csv'), index_col=0).PDB_IDS)

            if is_test_dataset_sampler:
                self.subset_pdb_code_set = set(test_cluster_meta.chain.str.slice(0, 4).to_list()) - possible_membrane_pdb_codes 
            else:
                self.subset_pdb_code_set = set(train_cluster_meta.chain.str.slice(0, 4).to_list()) - possible_membrane_pdb_codes

        # Maps sequence cluster to number of chains and vice versa
        self.chain_to_cluster = sequence_clusters.set_index('chain').to_dict()['cluster_representative']
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)
        self.subclusters_info = pd.read_pickle(params['subcluster_pickle_path'])
        
        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.train_split_clusters = set(train_cluster_meta['cluster_representative'].unique())
        self.test_split_clusters = set(test_cluster_meta['cluster_representative'].unique())
        self.cluster_to_chains, self.subclusters_info = self.filter_clusters(is_test_dataset_sampler, single_protein_debug)

        # Sample the first epoch.
        self.curr_samples = []
        self.sample_clusters()

        self.curr_batches = []
        self.construct_batches()

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return len(self.curr_batches)

    def filter_clusters(self, is_test_dataset_sampler: bool, single_protein_debug: bool) -> dict:
        """
        Filter clusters based on the given dataset sampler and the max protein length.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.
        """

        if is_test_dataset_sampler:
            curr_cluster_set = self.test_split_clusters
        else:
            curr_cluster_set = self.train_split_clusters
        
        if single_protein_debug:
            output = {}
            # output = {k: v for k,v in self.cluster_to_chains.items() if '4jnj_1-A-A' in v}
            n_copies = 1200 if not is_test_dataset_sampler else 100
            for idx in range(n_copies):
                # output[f'debug_{idx}-A-A'] = ['6w70_1-A-A']
                # output[f'debug_{idx}-A-A'] = ['4jnj_1-A-A']
                output[f'debug_{idx}-A-A'] = ['1x9q_1-A-A']
            return output

        if self.cluster_to_chains is None:
            raise NotImplementedError("Unreachable.")

        # Filter the clusters for train or test set.
        use_subset_pdb_codes = len(self.subset_pdb_code_set) > 0
        output = {k: v for k,v in self.cluster_to_chains.items() if k in curr_cluster_set}

        # If we don't have a max protein length, return the output.
        if self.max_protein_length is None:
            return output, filter_subclusters(self.subclusters_info, output)

        # Drop things that are longer than the max protein length and not in the subset pdb code set.
        filtered_output = defaultdict(list)
        for cluster_rep, cluster_list in output.items():
            for chain in cluster_list:
                if chain not in self.dataset.chain_key_to_index:
                    continue
                if (use_subset_pdb_codes and not (chain.split('_')[0] in self.subset_pdb_code_set)):
                    continue
                chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                if chain_len <= self.max_protein_length:
                    filtered_output[cluster_rep].append(chain)

        return filtered_output, filter_subclusters(self.subclusters_info, filtered_output)
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over the clusters and sample a random chain from each.
        for cluster, subclusters in self.subclusters_info.items():
            # Get a random subcluster.
            subcluster_sample_index = int(torch.randint(0, len(subclusters), (1,), generator=self.generator).item())
            subcluster_key = list(subclusters.keys())[subcluster_sample_index]

            # Get a random chain from the subcluster.
            subcluster_element_sampler_index = int(torch.randint(0, len(subclusters[subcluster_key]), (1,), generator=self.generator).item())
            subcluster_element_key = subclusters[subcluster_key][subcluster_element_sampler_index]

            self.curr_samples.append(self.dataset.chain_key_to_index[subcluster_element_key])

    def construct_batches(self):
        """
        Batches by size inspired by:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler
        """
        # Reset the current batches.
        self.curr_batches = []

        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        debug_sizes = []
        curr_list_sample_indices, curr_list_sizes = [], []
        for curr_size_sort_index in size_sort_indices:
            # Get current sample index and size.
            curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
            curr_size = sizes[curr_size_sort_index].item()

            # Add to the current batch if would not exceed batch size otherwise create a new batch.
            if sum(curr_list_sizes) + curr_size <= self.batch_size:
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
            else:
                # Add the current batch to the list of batches.
                self.curr_batches.append(curr_list_sample_indices)
                debug_sizes.append(sum(curr_list_sizes))

                # Reset the current batch.
                curr_list_sizes = [curr_size]
                curr_list_sample_indices = [curr_sample_index]

        # Store any remaining samples.
        if len(curr_list_sample_indices) > 0:
            self.curr_batches.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))

        # Shuffle the batches.
        if self.shuffle:
            shuffle_indices = torch.randperm(len(self.curr_batches), generator=self.generator).tolist()
            curr_batches_ = [self.curr_batches[x] for x in shuffle_indices]
            self.curr_batches = curr_batches_

        # Sanity check that we have the correct number of samples after iteration.
        assert sum(debug_sizes) == sizes.sum().item(), "Mismatch between number of samples and expected size of samples."

    def __iter__(self):
        # Yield the batches we created.
        for batch in self.curr_batches:
            yield batch

        # Resample for the next epoch, and create new batches.
        self.sample_clusters()
        self.construct_batches()


def _masked_sort_for_decoding_order(chain_mask, extra_atom_contact_mask, curr_batch_mask: Optional[torch.Tensor] = None):
    """
    Generates a permutation order for the indices of (N,) dim tensors that are True in curr_batch_mask.
    Randomly permutes indices in the order masks are added to ordered_mask_list.
    """
    ordered_mask_list = []

    if curr_batch_mask is None:
        curr_batch_mask = torch.ones_like(chain_mask, dtype=torch.bool)

    # First decode the provided residues in batch.order_mask (1 in chain_mask).
    ordered_mask_list.append(chain_mask & curr_batch_mask)
    # Decode residues that are 0 in chain chain mask but not in contact with random atoms.
    ordered_mask_list.append((~chain_mask) & (~extra_atom_contact_mask) & curr_batch_mask)
    # Decode residues in contact with random atoms not provided as sequence context last.
    ordered_mask_list.append(extra_atom_contact_mask & (~chain_mask) & curr_batch_mask)

    all_indices = []
    sort_keys = []
    for mask_idx, submask in enumerate(ordered_mask_list):
        # Get indices of residues in the current mask.
        indices = submask.nonzero().flatten()
        # Generate URNs offset by mask index to ensure sort preserves order in ordered_mask_list.
        rand_urns = torch.rand((submask.sum(),), device=chain_mask.device) + mask_idx
        all_indices.append(indices)
        sort_keys.append(rand_urns)
    all_indices = torch.cat(all_indices, dim=0)
    sort_keys = torch.cat(sort_keys, dim=0)

    # Sort the indices by the random keys.
    decoding_order = all_indices[sort_keys.argsort()]

    return decoding_order


@torch.no_grad()
def compute_kmeans_clusters(x, k, Niter=20):
    """
    Implements fast KMeans clustering in pytorch using the pykeops library.
        https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
    
    NOTE: could be precomputed but allows us to vary K for now.
    """

    if k < 1:
        raise ValueError("Need at least 1 cluster.")
    if Niter < 1:
        raise ValueError("Need at least 1 iteration.")

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:k, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, k, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c # type: ignore


class LigandMPNNDatasetSampler(Sampler):
    """
    Samples a single protein complex from precomputed mmseqs clusters.
    Ensures samples drawn evenly by sampling first from sequence clusters, then by pdb_code, then by assembly and chain.
    Iteration returns batched indices for use in UnclusteredProteinChainDataset.
    Pass to a DataLoader as a batch_sampler.
    """
    def __init__(
        self, dataset: UnclusteredProteinChainDataset, params: dict, is_train: bool, seed: Optional[int] = None, 
        max_protein_length: int = 10_000, subset_pdb_code_list: Optional[list] = None, soluble_proteins_only: bool = False
    ):
        # Set the random seed for reproducibility and consistent randomness between processes if parallelized.
        if seed is None:
            self.generator = torch.Generator(device='cpu')
        else:
            self.generator = torch.Generator().manual_seed(seed)

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.shuffle = params['sample_randomly']
        self.max_protein_length = max_protein_length
        self.subset_pdbs = subset_pdb_code_list
        self.soluble_proteins_only = soluble_proteins_only

        # Load the cluster data.
        sequence_clusters = pd.read_pickle(params['clustering_dataframe_path'])
        # if params['debug']:
        #     sequence_clusters = sequence_clusters[sequence_clusters.chain.str.find('w7') == 1]
        self.subclusters_info = pd.read_pickle(params['subcluster_pickle_path'])

        self.possible_membrane_pdb_codes = set(pd.read_csv(str(Path(__file__).parent.parent / 'files/membrane_excluded_PDBs.csv'), index_col=0).PDB_IDS)
        if self.soluble_proteins_only:
            self.representative_biological_assemblies = torch.load(Path(__file__).parent.parent / 'files/representative_bioa_set.pt', weights_only=False)
            self.pdb_chain_set_below_threshold = torch.load(Path(__file__).parent.parent / 'files/pdb_chain_set_below_threshold.pt', weights_only=False)

        self.is_train = is_train
        self.train_codes = ligandmpnn_training_pdb_codes
        self.val_codes = ligandmpnn_validation_pdb_codes
        self.test_codes = ligandmpnn_test_pdb_codes

        # Maps sequence cluster to number of chains and vice versa
        self.chain_to_cluster = sequence_clusters.set_index('chain').to_dict()['cluster_representative']
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)
        
        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.cluster_to_chains, self.subclusters_info = self.filter_clusters()

        # Sample for the first epoch, subsequent epochs will resample after iteration over samples is complete.
        self.curr_samples = []
        self.sample_clusters()

        self.curr_batches = []
        self.construct_batches()

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return len(self.curr_batches)

    def filter_clusters(self) -> Tuple[dict, dict]:
        """
        Filter clusters based on the given dataset sampler and the max protein length.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.
        """
        test_metal = ["1dwh", "1e4m", "1e6s", "1e72", "1f35", "1fee", "1job", "1lqk", "1m5e", "1m5f", "1moj", "1mxy", "1mxz", "1my1", "1nki", "1qum", "1sgf", "1t31", "1u3e", "2bdh", "2bx2", "2cfv", "2e6c", "2nq9", "2nqj", "2nz6", "2ou7", "2vxx", "2zwn", "3bvx", "3cv5", "3f4v", "3f5l", "3fgg", "3hg9", "3hkn", "3hkt", "3i9z", "3k7r", "3l24", "3l7t", "3m7p", "3mi9", "3o1u", "3u92", "3u93", "3u94", "3won", "4aoj", "4dy1", "4hzt", "4i0f", "4i0j", "4i0z", "4i11", "4i12", "4jd1", "4naz", "4wd8", "4x68", "5f55", "5f56", "5fgs", "5hez", "5i4j", "5l70", "5vde", "6a4x", "6buu", "6cyt", "6iv2", "6lkp", "6lrd", "6wdz", "6x75", "7dnr", "7e34", "7kii", "7n7g", "7s7l", "7s7m", "7w5e", "7wb2"]
        test_nucleic = ["1a0a", "1am9", "1an4", "1b01", "1bc7", "1bc8", "1di2", "1ec6", "1hlo", "1hlv", "1i3j", "1pvi", "1qum", "1sfu", "1u3e", "1xpx", "1yo5", "1zx4", "2c5r", "2c62", "2nq9", "2o4a", "2p5l", "2xdb", "2ypb", "2zhg", "2zio", "3adl", "3bsu", "3fc3", "3g73", "3gna", "3gx4", "3lsr", "3mj0", "3mva", "3n7q", "3olt", "3vok", "3vwb", "3zp5", "4ato", "4bhm", "4bqa", "4e0p", "4nid", "4wal", "5cm3", "5haw", "5mht", "5vc9", "5w9s", "5ybd", "6bjv", "6dnw", "6fqr", "6gdr", "6kbs", "6lff", "6lmj", "6od4", "6wdz", "6x70", "6y93", "7bca", "7c0g", "7el3", "7jsa", "7ju3", "7kii", "7kij", "7mtl", "7z0u", "8dwm"]

        val_code_set = set(self.val_codes)
        val_test_pdb_codes = set(self.val_codes + self.test_codes + test_metal + test_nucleic)

        # Get the cluster names containing any val or test pdb code information.
        contaminated_clusters = set()
        for cluster, pdb_code_list in self.cluster_to_chains.items():
            if any([x.split('_')[0] in val_test_pdb_codes for x in pdb_code_list]):
                contaminated_clusters.add(cluster)

        non_contam_clusters = {k: v for k,v in self.cluster_to_chains.items() if k not in contaminated_clusters}

        output = defaultdict(list)
        if self.is_train:
            for cluster_rep, cluster_list in non_contam_clusters.items():
                for chain in cluster_list:
                    if chain not in self.dataset.chain_key_to_index:
                        continue

                    if self.subset_pdbs is not None and chain.split('_')[0] not in self.subset_pdbs:
                        continue

                    # Not known membrane.
                    if self.soluble_proteins_only and (chain.split('_')[0] in self.possible_membrane_pdb_codes):
                        continue
                    
                    # Not correct bioa.
                    if self.soluble_proteins_only and not (chain.split('-')[0] in self.representative_biological_assemblies):
                        continue

                    # Not soluble enough
                    if self.soluble_proteins_only and not ((chain.split('-')[0], chain.split('-')[-1]) in self.pdb_chain_set_below_threshold):
                        continue

                    chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                    if chain_len <= self.max_protein_length:
                        output[cluster_rep].append(chain)
        else:
            potential_test_clusters = {k: v for k,v in self.cluster_to_chains.items() if k in contaminated_clusters}
            for cluster_rep, cluster_list in potential_test_clusters.items():
                for chain in cluster_list:
                    if chain not in self.dataset.chain_key_to_index or chain.split('_')[0] not in val_code_set:
                        continue

                    if self.subset_pdbs is not None and chain.split('_')[0] not in self.subset_pdbs:
                        continue

                    # Not known membrane.
                    if self.soluble_proteins_only and chain.split('_')[0] in self.possible_membrane_pdb_codes:
                        continue

                    # Not correct bioa.
                    if self.soluble_proteins_only and not (chain.split('-')[0] in self.representative_biological_assemblies):
                        continue

                    # Not soluble enough
                    if self.soluble_proteins_only and not ((chain.split('-')[0], chain.split('-')[-1]) in self.pdb_chain_set_below_threshold):
                        continue

                    chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                    if chain_len <= self.max_protein_length:
                        output[cluster_rep].append(chain)

        output2 = defaultdict(lambda: defaultdict(list))
        for cluster, subcluster_dict in self.subclusters_info.items():
            if cluster in output:
                for subcluster_centroid, subcluster_list in subcluster_dict.items():
                    for subcluster_entry in subcluster_list:
                        if subcluster_entry in output[cluster]:
                            output2[cluster][subcluster_centroid].append(subcluster_entry)

        return output, output2
        
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over mmseqs cluster and list of chains for that cluster.
        for cluster, subclusters in self.subclusters_info.items():
            # Get the subclusters to sample from.
            subcluster_sample_index = int(torch.randint(0, len(subclusters), (1,), generator=self.generator).item())
            subcluster_key = list(subclusters.keys())[subcluster_sample_index]

            # Sample a chain from the subcluster.
            subcluster_element_sampler_index = int(torch.randint(0, len(subclusters[subcluster_key]), (1,), generator=self.generator).item())
            subcluster_element_key = subclusters[subcluster_key][subcluster_element_sampler_index]

            # Yield the index of the sampled pdb_assembly-seg-chain.
            self.curr_samples.append(self.dataset.chain_key_to_index[subcluster_element_key])

    def construct_batches(self):
        """
        Batches by size inspired by:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler
        """
        # Reset the current batches.
        self.curr_batches = []

        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        debug_sizes = []
        curr_list_sample_indices, curr_list_sizes = [], []
        for curr_size_sort_index in size_sort_indices:
            # Get current sample index and size.
            curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
            curr_size = sizes[curr_size_sort_index].item()

            # Add to the current batch if would not exceed batch size otherwise create a new batch.
            if sum(curr_list_sizes) + curr_size <= self.batch_size:
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
            else:
                # Add the current batch to the list of batches.
                self.curr_batches.append(curr_list_sample_indices)
                debug_sizes.append(sum(curr_list_sizes))

                # Reset the current batch.
                curr_list_sizes = [curr_size]
                curr_list_sample_indices = [curr_sample_index]

        # Store any remaining samples.
        if len(curr_list_sample_indices) > 0:
            self.curr_batches.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))

        # Shuffle the batches.
        if self.shuffle:
            shuffle_indices = torch.randperm(len(self.curr_batches), generator=self.generator).tolist()
            curr_batches_ = [self.curr_batches[x] for x in shuffle_indices]
            self.curr_batches = curr_batches_

        # Sanity check that we have the correct number of samples after iteration.
        assert sum(debug_sizes) == sizes.sum().item(), "Mismatch between number of samples and expected size of samples."

    def __iter__(self):
        # Yield the batches we created.
        for batch in self.curr_batches:
            yield batch

        # Resample for the next epoch, and create new batches.
        self.sample_clusters()
        self.construct_batches()


def read_flat_csv(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split(',')
    return [x.lower() for x in lines]


class SolubleDatasetSampler(Sampler):
    def __init__(
        self, dataset: UnclusteredProteinChainDataset, params: dict, is_train: bool, seed: Optional[int] = None, 
        max_protein_length: int = 10_000, subset_pdb_code_list: Optional[list] = None
    ):
        # Set the random seed for reproducibility and consistent randomness between processes if parallelized.
        if seed is None:
            self.generator = torch.Generator(device='cpu')
        else:
            self.generator = torch.Generator().manual_seed(seed)

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.shuffle = params['sample_randomly']
        self.max_protein_length = max_protein_length
        self.subset_pdbs = subset_pdb_code_list

        # Load the cluster data.
        sequence_clusters = pd.read_pickle(params['clustering_dataframe_path'])
        self.subclusters_info = pd.read_pickle(params['subcluster_pickle_path'])

        self.membrane_pdb_codes = set(
            read_flat_csv('/home/nickhedu/programs/LASErMPNN/membrane_pdbs_chunk1.csv') + 
            read_flat_csv('/home/nickhedu/programs/LASErMPNN/membrane_pdbs_chunk2.csv')
        )

        self.is_train = is_train
        self.train_codes = ligandmpnn_training_pdb_codes
        self.val_codes = ligandmpnn_validation_pdb_codes
        self.test_codes = ligandmpnn_test_pdb_codes

        # Maps sequence cluster to number of chains and vice versa
        self.chain_to_cluster = sequence_clusters.set_index('chain').to_dict()['cluster_representative']
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)
        
        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.cluster_to_chains, self.subclusters_info = self.filter_clusters()

        # Sample for the first epoch, subsequent epochs will resample after iteration over samples is complete.
        self.curr_samples = []
        self.sample_clusters()

        self.curr_batches = []
        self.construct_batches()

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return len(self.curr_batches)

    def filter_clusters(self) -> Tuple[dict, dict]:
        """
        Filter clusters based on the given dataset sampler and the max protein length.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.

        """

        print(len(self.subclusters_info), "clusters before filtering for soluble proteins.")
        contaminated_clusters = []
        if self.is_train:
            output2 = defaultdict(lambda: defaultdict(list))
            for cluster, subcluster_dict in self.subclusters_info.items():
                if cluster in self.cluster_to_chains:
                    for subcluster_centroid, subcluster_list in subcluster_dict.items():
                        for subcluster_entry in subcluster_list:
                            # Ensure chain is in the dataset, not a membrane protein, and below max length.
                            if subcluster_entry in self.cluster_to_chains[cluster] and subcluster_entry in self.dataset.chain_key_to_index and not subcluster_entry.split('_')[0] in self.membrane_pdb_codes:
                                output2[cluster][subcluster_centroid].append(subcluster_entry)
                            elif subcluster_entry.split('_')[0] in self.membrane_pdb_codes:
                                contaminated_clusters.append(cluster)

            for cluster in contaminated_clusters:
                if cluster in output2:
                    del output2[cluster]
            
            print(len(output2), "clusters after filtering for soluble proteins.")
            return self.cluster_to_chains, output2
        else:
            # No test set.
            return  {}, {}
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over mmseqs cluster and list of chains for that cluster.
        for cluster, subclusters in self.subclusters_info.items():
            # Get the subclusters to sample from.
            subcluster_sample_index = int(torch.randint(0, len(subclusters), (1,), generator=self.generator).item())
            subcluster_key = list(subclusters.keys())[subcluster_sample_index]

            # Sample a chain from the subcluster.
            subcluster_element_sampler_index = int(torch.randint(0, len(subclusters[subcluster_key]), (1,), generator=self.generator).item())
            subcluster_element_key = subclusters[subcluster_key][subcluster_element_sampler_index]

            # Yield the index of the sampled pdb_assembly-seg-chain.
            self.curr_samples.append(self.dataset.chain_key_to_index[subcluster_element_key])

    def construct_batches(self):
        """
        Batches by size inspired by:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler
        """
        # Reset the current batches.
        self.curr_batches = []

        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        debug_sizes = []
        curr_list_sample_indices, curr_list_sizes = [], []
        for curr_size_sort_index in size_sort_indices:
            # Get current sample index and size.
            curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
            curr_size = sizes[curr_size_sort_index].item()

            # Add to the current batch if would not exceed batch size otherwise create a new batch.
            if sum(curr_list_sizes) + curr_size <= self.batch_size:
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
            else:
                # Add the current batch to the list of batches.
                self.curr_batches.append(curr_list_sample_indices)
                debug_sizes.append(sum(curr_list_sizes))

                # Reset the current batch.
                curr_list_sizes = [curr_size]
                curr_list_sample_indices = [curr_sample_index]

        # Store any remaining samples.
        if len(curr_list_sample_indices) > 0:
            self.curr_batches.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))

        # Shuffle the batches.
        if self.shuffle:
            shuffle_indices = torch.randperm(len(self.curr_batches), generator=self.generator).tolist()
            curr_batches_ = [self.curr_batches[x] for x in shuffle_indices]
            self.curr_batches = curr_batches_

        # Sanity check that we have the correct number of samples after iteration.
        assert sum(debug_sizes) == sizes.sum().item(), "Mismatch between number of samples and expected size of samples."

    def __iter__(self):
        # Yield the batches we created.
        for batch in self.curr_batches:
            yield batch

        # Resample for the next epoch, and create new batches.
        self.sample_clusters()
        self.construct_batches()


class JerryDatasetSampler(Sampler):
    def __init__(
        self, dataset: UnclusteredProteinChainDataset, params: dict, is_train: bool, seed: Optional[int] = None, 
        max_protein_length: int = 10_000, subset_pdb_code_list: Optional[list] = None
    ):
        # Set the random seed for reproducibility and consistent randomness between processes if parallelized.
        if seed is None:
            self.generator = torch.Generator(device='cpu')
        else:
            self.generator = torch.Generator().manual_seed(seed)

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.shuffle = params['sample_randomly']
        self.max_protein_length = max_protein_length
        self.subset_pdbs = subset_pdb_code_list

        # Load the cluster data.
        sequence_clusters = pd.read_pickle(params['clustering_dataframe_path'])
        self.subclusters_info = pd.read_pickle(params['subcluster_pickle_path'])

        # self.membrane_pdb_codes = set(
        #     read_flat_csv('/home/nickhedu/programs/LASErMPNN/membrane_pdbs_chunk1.csv') + 
        #     read_flat_csv('/home/nickhedu/programs/LASErMPNN/membrane_pdbs_chunk2.csv')
        # )

        self.membrane_pdb_codes = set(
            read_flat_csv('/home/nickhedu/programs/LASErMPNN/3wfh_combine_list.txt')
        )

        self.is_train = is_train
        self.train_codes = ligandmpnn_training_pdb_codes
        self.val_codes = ligandmpnn_validation_pdb_codes
        self.test_codes = ligandmpnn_test_pdb_codes

        # Maps sequence cluster to number of chains and vice versa
        self.chain_to_cluster = sequence_clusters.set_index('chain').to_dict()['cluster_representative']
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)
        
        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.cluster_to_chains, self.subclusters_info = self.filter_clusters()

        # Sample for the first epoch, subsequent epochs will resample after iteration over samples is complete.
        self.curr_samples = []
        self.sample_clusters()

        self.curr_batches = []
        self.construct_batches()

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return len(self.curr_batches)

    def filter_clusters(self) -> Tuple[dict, dict]:
        """
        Filter clusters based on the given dataset sampler and the max protein length.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.

        """

        print(len(self.subclusters_info), "clusters before filtering for soluble proteins.")
        contaminated_clusters = []
        if self.is_train:
            output2 = defaultdict(lambda: defaultdict(list))
            for cluster, subcluster_dict in self.subclusters_info.items():
                if cluster in self.cluster_to_chains:
                    for subcluster_centroid, subcluster_list in subcluster_dict.items():
                        for subcluster_entry in subcluster_list:
                            # Ensure chain is in the dataset, not a membrane protein, and below max length.
                            if subcluster_entry in self.cluster_to_chains[cluster] and subcluster_entry in self.dataset.chain_key_to_index and not subcluster_entry.split('_')[0] in self.membrane_pdb_codes:
                                output2[cluster][subcluster_centroid].append(subcluster_entry)
                            elif subcluster_entry.split('_')[0] in self.membrane_pdb_codes:
                                contaminated_clusters.append(cluster)

            for cluster in contaminated_clusters:
                if cluster in output2:
                    del output2[cluster]
            
            print(len(output2), "clusters after filtering for soluble proteins.")
            return self.cluster_to_chains, output2
        else:
            # No test set.
            return  {}, {}
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over mmseqs cluster and list of chains for that cluster.
        for cluster, subclusters in self.subclusters_info.items():
            # Get the subclusters to sample from.
            subcluster_sample_index = int(torch.randint(0, len(subclusters), (1,), generator=self.generator).item())
            subcluster_key = list(subclusters.keys())[subcluster_sample_index]

            # Sample a chain from the subcluster.
            subcluster_element_sampler_index = int(torch.randint(0, len(subclusters[subcluster_key]), (1,), generator=self.generator).item())
            subcluster_element_key = subclusters[subcluster_key][subcluster_element_sampler_index]

            # Yield the index of the sampled pdb_assembly-seg-chain.
            self.curr_samples.append(self.dataset.chain_key_to_index[subcluster_element_key])

    def construct_batches(self):
        """
        Batches by size inspired by:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler
        """
        # Reset the current batches.
        self.curr_batches = []

        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        debug_sizes = []
        curr_list_sample_indices, curr_list_sizes = [], []
        for curr_size_sort_index in size_sort_indices:
            # Get current sample index and size.
            curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
            curr_size = sizes[curr_size_sort_index].item()

            # Add to the current batch if would not exceed batch size otherwise create a new batch.
            if sum(curr_list_sizes) + curr_size <= self.batch_size:
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
            else:
                # Add the current batch to the list of batches.
                self.curr_batches.append(curr_list_sample_indices)
                debug_sizes.append(sum(curr_list_sizes))

                # Reset the current batch.
                curr_list_sizes = [curr_size]
                curr_list_sample_indices = [curr_sample_index]

        # Store any remaining samples.
        if len(curr_list_sample_indices) > 0:
            self.curr_batches.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))

        # Shuffle the batches.
        if self.shuffle:
            shuffle_indices = torch.randperm(len(self.curr_batches), generator=self.generator).tolist()
            curr_batches_ = [self.curr_batches[x] for x in shuffle_indices]
            self.curr_batches = curr_batches_

        # Sanity check that we have the correct number of samples after iteration.
        assert sum(debug_sizes) == sizes.sum().item(), "Mismatch between number of samples and expected size of samples."

    def __iter__(self):
        # Yield the batches we created.
        for batch in self.curr_batches:
            yield batch

        # Resample for the next epoch, and create new batches.
        self.sample_clusters()
        self.construct_batches()