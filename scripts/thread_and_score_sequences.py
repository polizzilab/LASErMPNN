#!/usr/bin/env python

"""
MIT License

Inference script for threading sequences onto a backbone structure using LASErMPNN. 
Given a fasta file of sequences and a backbone pdb file, this script will output pdb 
files of the fasta sequences threaded onto the backbone with LASErMPNN predicted rotamers.

Unconditional probabilities of the threaded sequences will be computed and saved to a csv file 
in the output directory as well as written to the b-factors of the output pdb files.

Benjamin Fry (bfry@g.harvard.edu)
"""

import os
import sys
import argparse
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import pandas as pd
import prody as pr
from tqdm import tqdm

LASErMPNN_INSTALL_DIR = Path(__file__).absolute().parent
sys.path.append(str(LASErMPNN_INSTALL_DIR.parent.parent))

from LASErMPNN.utils.constants import aa_short_to_long, aa_short_to_idx
from LASErMPNN.run_inference import (
    ProteinComplexData, load_model_from_parameter_dict, sample_model,
    output_protein_structure, output_ligand_structure
)


def cli():
    default_weights = Path(__file__).absolute().parent.parent / 'model_weights' / 'laser_weights_0p1A_nothing_heldout.pt'
    parser = argparse.ArgumentParser(description='Given a fasta file of sequences, uses LASErMPNN as a packer outputting the input fasta sequences threaded onto the input backbone as pdb files.')
    parser.add_argument('fasta', type=str, help='Path to the fasta file of sequences to be threaded onto the input backbone.')
    parser.add_argument('backbone', type=str, help='Path to the pdb file of the backbone structure to thread the sequences onto.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where the threaded pdb files will be saved.')
    parser.add_argument('-w', '--weights', type=str, default=default_weights, help='Path to the weights file for LASErMPNN. If not provided, default weights will be used.')
    main(**vars(parser.parse_args()))


def parse_threading_fasta(fasta_file: os.PathLike, expected_length: int) -> List[str]:
    """
    Parses the sequences out of a fasta file. Returns a list of sequence. Raises an error if any of the sequences are not of the expected length.
    """
    sequences = []
    with open(fasta_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('>'):
                continue
            sequence = line.strip()

            if len(sequence) != expected_length:
                raise ValueError(f"Sequence {sequence} is not of the expected length {expected_length}.")
            sequences.append(sequence)
    return sequences


def make_all_gly_backbone(sequence: str, ref_prot_: pr.AtomGroup) -> pr.AtomGroup:
    ref_prot = deepcopy(ref_prot_)
    final_ag = None
    for idx, res_ in enumerate(ref_prot.select('same residue as name CA').copy().iterResidues()):
        res_.setResname(aa_short_to_long.get(sequence[idx], 'XAA'))
        if final_ag is None:
            final_ag = res_.select('name N CA C O').copy()
        else:
            final_ag += res_.select('name N CA C O').copy()

    final_ag += ref_prot.select('not (same residue as name CA)').copy()
    final_ag.setTitle(f"Threaded sequence: {sequence}")

    # Force prody to compute and cache the reserved 'protein'/'aminoacid' flags
    # so the keys exist in the AtomGroup's _flags dict. These are reserved
    # (cannot be setFlags-ed), and are populated lazily; get_all_gly_protein()
    # downstream reads ag._flags['protein'] directly, which KeyErrors if the
    # flags were never accessed. The cached keys propagate through the copy and
    # concatenation done there.
    final_ag.getFlags('protein')
    final_ag.getFlags('aminoacid')

    return final_ag


@torch.no_grad()
def main(fasta, backbone, output_dir, weights):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    sequence_nll_csv = output_dir / 'sequence_nlls.csv'

    assert Path(weights).absolute().exists(), f"Weight file {weights} does not exist. Please provide a valid path to the weights file."

    print(f"Threading sequences from {fasta} onto backbone {backbone} and saving to {output_dir}")

    backbone_prot = pr.parsePDB(backbone)
    num_ca_carbons = len(backbone_prot.select('name CA').getCoords())
    sequences = parse_threading_fasta(fasta, num_ca_carbons)
    model, params = load_model_from_parameter_dict(weights, 'cpu', strict=True)

    sequence_to_nll_data = []
    for idx, sequence in enumerate(tqdm(sequences, total=len(sequences), desc="Threading sequences")):
        threaded = make_all_gly_backbone(sequence, backbone_prot)
        protein_hv = threaded.getHierView()

        data = ProteinComplexData(
            protein_hv, f'threaded_{idx}', 
            treat_noncanonical_as_ligand=False, 
            first_shell_ca_distance=10.0,
            first_shell_buried_only=True,
            first_shell_burial_calc_hull_alpha=14.0
        )
        batch_data = data.output_batch_data(False)
        batch_data.chain_mask = torch.ones_like(batch_data.chain_mask)
        sampled_output = sample_model(
            model, batch_data, 1e-6, 0.0, params, False, 1e-6, 
            repack_all=True, 
            budget_residue_mask=None,
            disable_charged_fs=False,
            disable_pbar=True
        )

        sampled_probs = sampled_output.sequence_logits.softmax(dim=-1).gather(1, sampled_output.sampled_sequence_indices.unsqueeze(-1)).squeeze(-1)
        full_atom_coords = model.rotamer_builder.build_rotamers(batch_data.backbone_coords, sampled_output.sampled_chi_degrees, sampled_output.sampled_sequence_indices, add_nonrotatable_hydrogens=True)
        nh_coords = model.rotamer_builder.impute_backbone_nh_coords(full_atom_coords.float(), sampled_output.sampled_sequence_indices, batch_data.phi_psi_angles[:, 0].unsqueeze(-1))
        full_atom_coords = model.rotamer_builder.cleanup_titratable_hydrogens(full_atom_coords.float(), sampled_output.sampled_sequence_indices, nh_coords, batch_data, model.hbond_network_detector) # type: ignore

        out_prot = output_protein_structure(full_atom_coords, sampled_output.sampled_sequence_indices, data.residue_identifiers, nh_coords, sampled_probs)
        try:
            out_lig = output_ligand_structure(data.ligand_info)
            out_prot += out_lig
        except:
            pass
        out_prot.getFlags('protein')
        out_prot.getFlags('aminoacid')

        ###########
        # Score with unconditional probs.
        protein_hv = out_prot.getHierView()
        data = ProteinComplexData(protein_hv, f"threaded_{idx}")
        batch_data = data.output_batch_data(fix_beta=False)
        batch_data.to_device('cpu')
        batch_data.construct_graphs(
            model.rotamer_builder,
            model.ligand_featurizer,
            **params['model_params']['graph_structure'],
            protein_training_noise = 0.0,
            ligand_training_noise = 0.0,
            subgraph_only_dropout_rate = 0.0,
            num_adjacent_residues_to_drop=1,
            build_hydrogens = params['model_params']['build_hydrogens'],
        )
        batch_data.generate_decoding_order(False)
        fs_mask = batch_data.first_shell_ligand_contact_mask.cpu()
        sequence_logits, *_ = model.forward(batch_data, return_unconditional_probabilities=True)
        probs = sequence_logits.softmax(dim=-1).cpu()
        ###########

        # Write the unconditional probabilities to the b-factors of the threaded pdb file and write pdb file to disk.
        unconditional_probs = probs.gather(-1, torch.tensor([aa_short_to_idx[x] for x in sequence]).unsqueeze(-1)).squeeze(-1)
        out_prot = output_protein_structure(full_atom_coords, sampled_output.sampled_sequence_indices, data.residue_identifiers, nh_coords, unconditional_probs)
        try:
            out_lig = output_ligand_structure(data.ligand_info)
            out_prot += out_lig
        except:
            pass
        opath = output_dir / f"threaded_{idx}.pdb" 
        pr.writePDB(str(opath.absolute()), out_prot)

        sequence_nll = -1 * unconditional_probs.log10().mean().item()
        sequence_to_nll_data.append({'sequence': sequence, 'unconditional_nll': sequence_nll})

    pd.DataFrame(sequence_to_nll_data).to_csv(sequence_nll_csv, index=False)


if __name__ == '__main__':
    cli()
