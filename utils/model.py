import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_min, scatter_max, scatter_log_softmax, scatter_softmax
from torch_cluster import radius_graph
from tqdm import tqdm
from copy import deepcopy

from .build_rotamers import RotamerBuilder
from .ligand_featurization import LigandFeaturizer
from .hbond_network import RigorousHydrogenBondNetworkDetector
from .pdb_dataset import BatchData, LigandData
from .model_generics import HomoGATv2, HeteroGATv2, DenseMLP, DenseGVP, GVP, EquivariantData, EquivariantLayerNorm, _norm_no_nan
from .constants import aa_short_to_idx
from .spice_dataset import SpiceBatchData, POSSIBLE_HYBRIDIZATION_LIST, POSSIBLE_FORMAL_CHARGE_LIST, POSSIBLE_NUM_HYDROGENS_LIST, POSSIBLE_DEGREE_LIST, POSSIBLE_IS_AROMATIC_LIST
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class Sampled_Output:
    """
    Stores the output of a sample from the model.
    """
    sequence_logits: torch.Tensor
    chi_logits: torch.Tensor
    sampled_sequence_indices: torch.Tensor
    sampled_chi_encoding: torch.Tensor
    sampled_chi_degrees: torch.Tensor

    def to(self, device: torch.device) -> 'Sampled_Output':
        return Sampled_Output(self.sequence_logits.to(device), self.chi_logits.to(device), self.sampled_sequence_indices.to(device), self.sampled_chi_encoding.to(device), self.sampled_chi_degrees.to(device))

def create_sampling_output(num_residues: int, num_chi_bins: int, device: torch.device) -> Sampled_Output:
    """
    Initializes the output tensors to zeros.
    """
    sequence_logits = torch.zeros((num_residues, 21), device=device)
    sampled_sequence_indices = torch.full((num_residues,), 21, dtype=torch.long, device=device)

    chi_logits = torch.zeros((num_residues, 4, num_chi_bins), device=device)
    chi_encoding = torch.full((num_residues, 4, num_chi_bins), torch.nan, device=device)
    chi_degrees = torch.full((num_residues, 4), torch.nan, device=device)

    return Sampled_Output(sequence_logits, chi_logits, sampled_sequence_indices, chi_encoding, chi_degrees)


def minp_warp_logits(logits: torch.Tensor, min_p: float, min_indices_to_keep: int = 1) -> torch.Tensor:
    """
    Given an input tensor [B, N] of logits and a threshold minimum probability fraction min_p, 
        under which all probabilities are set to 0. The min_p probability threshold is scaled 
        by the top probability of each sequence in the batch so the threshold is
        relative to the top token's probability.  
    
    Returns a tensor of the same shape with pre-softmax logits set to -inf for softmax.

    Adapted from: 
        https://github.com/menhguin/minp_paper/blob/main/implementation
    """
    # Setting min_p to 0.0 will return the original logits.
    if min_p == 0.0:
        return logits

    # Convert logits to probabilities
    probs = logits.softmax(dim=-1)
    # Get the probability of the top token for each sequence in the batch
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    # Calculate the actual min_p threshold by scaling min_p with the top token's probability
    scaled_min_p = min_p * top_probs
    # Mask out the probabilities below the threshold
    indices_to_remove = probs < scaled_min_p
    sorted_indices = probs.argsort(dim=-1, descending=True)
    sorted_indices_to_remove = torch.gather(indices_to_remove, dim=-1, index=sorted_indices)
    # Keep at least min_indices_to_keep indices
    sorted_indices_to_remove[:, :min_indices_to_keep] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    # Mask out the probabilities below the threshold
    scores_processed = logits.masked_fill(indices_to_remove, float('-Inf'))
    return scores_processed


def compute_frame_vector_ligand_displacement_vector_dot_product(lig_coords, backbone_coords, backbone_vectors, lig_pr_edge_index):
    # Compute the vector pointing from the ca to the ligand atom and normalize it.
    ligand_displacement_vec = lig_coords[lig_pr_edge_index[0]] - backbone_coords[lig_pr_edge_index[1], 1]
    ligand_displacement_vec = ligand_displacement_vec / _norm_no_nan(ligand_displacement_vec, axis=-1, keepdims=True)
    normalized_backbone_vecs = backbone_vectors / _norm_no_nan(backbone_vectors, axis=-1, keepdims=True)

    dot_prods = torch.einsum('ijk,ik->ij', normalized_backbone_vecs[lig_pr_edge_index[1]], ligand_displacement_vec)
    return dot_prods


def compute_protein_frame_vector_dot_product(backbone_vectors, pr_pr_edge_index):
    # B x N x 3
    normalized_backbone_vecs = backbone_vectors / _norm_no_nan(backbone_vectors, axis=-1, keepdims=True)

    # (B x 1 X N x 3) , (B, N, 1, 3) -> (B, 1, N, N, 1)
    dot_prods = torch.einsum(
        'bijk,bmnk->bijmn', 
        normalized_backbone_vecs[pr_pr_edge_index[0]].unsqueeze(1), 
        normalized_backbone_vecs[pr_pr_edge_index[1]].unsqueeze(2)
    ).flatten(start_dim=1)

    # B, N**2
    return dot_prods


class LASErMPNN(nn.Module):
    def __init__(
        self, 
        ligand_encoder_params: dict, node_embedding_dim: int, protein_edge_embedding_dim: int,
        chi_angle_rbf_bin_width: int, prot_prot_edge_rbf_params: dict, lig_prot_edge_rbf_params: dict, 
        num_encoder_layers: int, num_decoder_layers: int, additional_ligand_mlp: bool, 
        num_laser_vectors: int, **kwargs
    ):
        super(LASErMPNN, self).__init__()

        self.num_vectors = num_laser_vectors
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = protein_edge_embedding_dim
        self.ligand_encoder_params = ligand_encoder_params

        self.hbond_network_detector = RigorousHydrogenBondNetworkDetector()
        self.rotamer_builder = RotamerBuilder(chi_angle_rbf_bin_width)
        self.chi_embedding_dim = self.rotamer_builder.num_chi_bins * 4
        self.ligand_featurizer = LigandFeaturizer(**kwargs)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.num_ligand_encoder_vectors = ligand_encoder_params['model_params']['num_ligand_encoder_vectors']
        self.ligand_encoder = LigandEncoderModule(
            cg_node_in_dim=self.ligand_featurizer.output_dim, 
            **ligand_encoder_params['model_params'], 
            edge_embedding_dim=ligand_encoder_params['model_params']['ligand_edge_embedding_dim'],
            num_vectors=self.num_ligand_encoder_vectors,
        )

        # If additional_ligand_mlp is True, then the ligand encoder output will be passed through an MLP before being used in the protein encoder, should help learn from pretraining.
        if additional_ligand_mlp:
            self.ligand_encoder_output_gvp = DenseGVP(
                (ligand_encoder_params['model_params']['node_embedding_dim'], self.num_ligand_encoder_vectors), 
                (node_embedding_dim, self.num_ligand_encoder_vectors), 
                (node_embedding_dim, 0), 
                dropout=kwargs['dropout'], 
                intermediate_norm=True
            )
        else:
            assert ligand_encoder_params['model_params']['node_embedding_dim'] == node_embedding_dim, "Ligand encoder node embedding dim must match protein node embedding dim."
        

        self.protein_encoder_layers = nn.ModuleList([
            LASErMPNN_Encoder(node_embedding_dim, protein_edge_embedding_dim, num_vectors=num_laser_vectors, **kwargs)
            for _ in range(num_encoder_layers)
        ])
        # self, node_embedding_dim: int, edge_embedding_dim: int, chi_embedding_dim: int, atten_head_aggr_layers: int, num_attention_heads: int, dropout: float, **kwargs
        self.protein_decoder_layers = nn.ModuleList([
            LASErMPNN_Decoder(
                node_embedding_dim=node_embedding_dim, 
                edge_embedding_dim=protein_edge_embedding_dim, 
                chi_embedding_dim=self.chi_embedding_dim, 
                num_vectors=num_laser_vectors, 
                **kwargs
            ) for _ in range(num_decoder_layers)
        ])

        self.chi_prediction_layers = nn.ModuleList([
            DenseMLP(
                input_dim = (2 * node_embedding_dim) + (idx * self.rotamer_builder.num_chi_bins), 
                latent_dim = node_embedding_dim, 
                output_dim = self.rotamer_builder.num_chi_bins, 
                mlp_dropout = kwargs['dropout']
            ) for idx in range(4)
        ])

        self.chi_offset_prediction_layers = nn.ModuleList([
            DenseMLP(
                input_dim = 2 * self.rotamer_builder.num_chi_bins, 
                latent_dim = node_embedding_dim, 
                output_dim = 1, 
                mlp_dropout = kwargs['dropout']
            ) for _ in range(4)
        ])

        self.chi_vector_update_layers = nn.ModuleList([
            GVP(
                (node_embedding_dim + ((idx + 1) * self.rotamer_builder.num_chi_bins), num_laser_vectors),
                (node_embedding_dim, num_laser_vectors),
                vector_gate=True
            ) for idx in range(3) # Only need 3 updates since dont use prot nodes after last chi angle is decoded.
        ])

        self.chi_vector_layer_norms = nn.ModuleList([
            EquivariantLayerNorm((node_embedding_dim, num_laser_vectors), vector_only=True) for _ in range(3)
        ])

        self.prot_prot_rbf_encoding = RBF_Encoding(**prot_prot_edge_rbf_params)
        self.lig_prot_rbf_encoding = RBF_Encoding(**lig_prot_edge_rbf_params)
        self.prot_prot_edge_input_layer = nn.Linear((self.prot_prot_rbf_encoding.num_bins * 25) + (self.num_vectors ** 2), protein_edge_embedding_dim)
        self.lig_prot_edge_input_layer = nn.Linear((self.lig_prot_rbf_encoding.num_bins * 5) + self.num_vectors, protein_edge_embedding_dim)

        self.backbone_frame_vec_input_layer = GVP((0, 4), (0, num_laser_vectors))
        self.backbone_frame_vec_norm = EquivariantLayerNorm((0, num_laser_vectors), vector_only=True)

        # 20 AAs + X + NotDecoded
        self.sequence_label_embedding = nn.Embedding(22, node_embedding_dim)
        self.sequence_output_layer = nn.Linear(node_embedding_dim, 21)

        self.gelu = nn.GELU()

    @property
    def device(self) -> torch.device:
        """
        Returns the device that the model is currently on when addressed as model.device
        """
        return next(self.parameters()).device
    
    def apply_encoding_layers(self, batch: BatchData) -> Tuple[EquivariantData, EquivariantData, torch.Tensor, torch.Tensor]:
        """
        """
        assert batch.ligand_data is not None, "Ligand data must be defined in batch data, even if filled with empty tensors."
        # Use a HomoGATv2 to encode the ligand nodes if they exist.
        cg_node_equidata = self.ligand_encoder(batch.ligand_data)

        if hasattr(self, 'ligand_encoder_output_gvp'):
            cg_node_equidata = self.ligand_encoder_output_gvp(cg_node_equidata)
        
        # Initialize protein nodes to zeros, representations will be built-up by encoding process.
        prot_nodes = torch.zeros((batch.num_residues, self.node_embedding_dim), device=self.device)

        # initialize protein node vectors to be vectors pointing between backbone frame atoms.
        n_ca_vec = batch.backbone_coords[:, 0] - batch.backbone_coords[:, 1]
        ca_c_vec = batch.backbone_coords[:, 3] - batch.backbone_coords[:, 1]
        n_ca_vec = n_ca_vec / n_ca_vec.norm(dim=-1, keepdim=True)
        ca_c_vec = ca_c_vec / ca_c_vec.norm(dim=-1, keepdim=True)
        frame_bisector = -1 * (n_ca_vec + ca_c_vec) / torch.norm(n_ca_vec + ca_c_vec, dim=-1, keepdim=True)
        ca_to_cb = (batch.backbone_coords[:, 2] - batch.backbone_coords[:, 1]).unsqueeze(1)
        ca_to_cb = ca_to_cb / ca_to_cb.norm(dim=-1, keepdim=True)

        prot_frame_vecs = torch.cat([frame_bisector.unsqueeze(1), ca_to_cb, n_ca_vec.unsqueeze(1), ca_c_vec.unsqueeze(1)], dim=1)

        # NOTE: prot_nodes remains zeros after passing through the GVP layer (see test_model_generics.py) 
        #    so we can use it to pass EquivariantData to the GVP layer.
        prot_equidata = EquivariantData(prot_nodes, prot_frame_vecs)
        prot_equidata = self.backbone_frame_vec_norm(self.backbone_frame_vec_input_layer(prot_equidata))

        # Compute angular information for protein-ligand edges with dot product of frame vectors and ligand displacement vectors.
        ligand_displacement_dot_products = compute_frame_vector_ligand_displacement_vector_dot_product(batch.ligand_data.lig_coords, batch.backbone_coords, prot_equidata.vectors, batch.lig_pr_edge_index)
        frame_vector_dot_products = compute_protein_frame_vector_dot_product(prot_equidata.vectors, batch.pr_pr_edge_index)

        # Encode protein nodes and edges with Heterograph-GAT-Augmented-ProteinMPNN update layers.
        pr_pr_eattr = self.prot_prot_edge_input_layer(torch.cat([self.prot_prot_rbf_encoding(batch.pr_pr_edge_distance).flatten(start_dim=1), frame_vector_dot_products], dim=1))
        lig_pr_eattr = self.lig_prot_edge_input_layer(torch.cat([self.lig_prot_rbf_encoding(batch.lig_pr_edge_distance).flatten(start_dim=1), ligand_displacement_dot_products], dim=1))

        # if self.training:

        # Apply protein heterograph encoder layers.
        for enc_layer in self.protein_encoder_layers:
            prot_equidata, pr_pr_eattr, lig_pr_eattr = enc_layer(prot_equidata, cg_node_equidata, pr_pr_eattr, lig_pr_eattr, batch)

        return cg_node_equidata, prot_equidata, pr_pr_eattr, lig_pr_eattr
    
    def get_logits_for_score(self, batch: BatchData, decoding_order, sequence_indices, chi_angles, decoding_order_generator: Optional[nn.Module] = None):
        """
        Very similar to forward pass, but we want to score (get probabilities for) different sequences, chi_angles, and decoding orders.
        TODO: extract applying decoding layers with teacher-forcing to a separate function to avoid code reuse.

        If Decoding order is not provided, we will generate one from the encoder embeddings.
        """
        assert batch.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."

        lig_nodes, prot_nodes, pr_pr_eattr, lig_pr_eattr = self.apply_encoding_layers(batch)
        decoding_order_sort_indices = decoding_order.argsort()

        pr_pr_edge_mask = (decoding_order_sort_indices[batch.pr_pr_edge_index[0]] < decoding_order_sort_indices[batch.pr_pr_edge_index[1]])

        unmasked_edge_indices = batch.pr_pr_edge_index[:, pr_pr_edge_mask]
        unmasked_edge_attrs = pr_pr_eattr[pr_pr_edge_mask]
        masked_edge_indices = batch.pr_pr_edge_index[:, ~pr_pr_edge_mask]
        masked_edge_attrs = pr_pr_eattr[~pr_pr_edge_mask]

        # Use ground-truth chi angles and sequence labels to build edge features for teacher forcing.
        chi_angle_encoding = self.rotamer_builder.compute_binned_degree_basis_function(chi_angles).nan_to_num()
        binned_degree_chi_for_teacher_forcing = chi_angle_encoding.flatten(start_dim=1)[unmasked_edge_indices[0]]
        sequence_embedding_nodes = self.sequence_label_embedding(sequence_indices)
        sequence_embedding_for_teacher_forcing = sequence_embedding_nodes[unmasked_edge_indices[0]]

        # Use decoding order mask to selectively provide label data for nodes that have been previously decoded.
        source_node_edge_features_exp_unmasked = torch.cat([sequence_embedding_for_teacher_forcing, binned_degree_chi_for_teacher_forcing, unmasked_edge_attrs], dim=1)
        masked_sequence_embeddings = self.sequence_label_embedding(torch.full_like(sequence_indices, fill_value=21))
        masked_chi_embedding = torch.zeros(masked_edge_attrs.shape[0], binned_degree_chi_for_teacher_forcing.shape[1], device=batch.device)
        source_node_edge_features_exp_masked = torch.cat([masked_sequence_embeddings[masked_edge_indices[0]], masked_chi_embedding, masked_edge_attrs, prot_nodes.scalars[masked_edge_indices[0]]], dim=1)

        encoder_nodes = prot_nodes.scalars.detach().clone()
        encoder_eattrs = pr_pr_eattr.detach().clone()

        # Iteratively update protein nodes with teacher-forced decoding.
        for didx, decoder_layer in enumerate(self.protein_decoder_layers):
            # Provides node features for unmasked source nodes at current step of decoding.
            curr_nodes_exp = prot_nodes.scalars[unmasked_edge_indices[0]]
            curr_edge_features_unmasked = torch.cat([source_node_edge_features_exp_unmasked, curr_nodes_exp], dim=1)

            # Add self-edges to nodes that have not yet been decoded.
            all_edge_features = torch.cat([source_node_edge_features_exp_masked, curr_edge_features_unmasked], dim=0)
            all_edge_indices = torch.cat([masked_edge_indices, unmasked_edge_indices], dim=1)

            # Update node embeddings with teacher-forced edges.
            prot_nodes = decoder_layer(all_edge_features, prot_nodes, all_edge_indices, lig_nodes, lig_pr_eattr, batch.lig_pr_edge_index)

        # Sequence logits generate sequence log-probs.
        sequence_logits = self.sequence_output_layer(prot_nodes.scalars)

        # Compute log-probs from the given decoding order
        if decoding_order_generator is None:
            decoding_order_log_probs = torch.ones_like(decoding_order).float()
        else:
            # decoding_order_log_probs = decoding_order_generator(prot_nodes.clone_detach(), decoding_order, batch.batch_indices)
            decoding_order_log_probs = decoding_order_generator(prot_nodes, decoding_order, batch.batch_indices)

        # Generate chi-logits.
        prot_scalars = prot_nodes.scalars
        sampled_angle_mask = ~chi_angles.isnan()
        output_chi_logits = torch.full((batch.num_residues, 4, self.rotamer_builder.num_chi_bins), fill_value=torch.nan, device=self.device, dtype=prot_scalars.dtype)
        prev_chi = torch.empty((batch.num_residues, 0), device=self.device, dtype=prot_scalars.dtype)
        for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
            # Predict a chi angle from final protein node embeddings.
            chi_logits = chi_layer(torch.cat([prot_scalars, sequence_embedding_nodes, prev_chi], dim=1))

            # # Store sampled angles and logits in output tensors.
            curr_masks = sampled_angle_mask[:, chi_idx]
            output_chi_logits[curr_masks, chi_idx] = chi_logits[curr_masks]

            # Concatenate previously decoded chi angles to provide input to next chi angle prediction layer.
            prev_chi = chi_angle_encoding[:, :chi_idx + 1].flatten(start_dim=1)

            # Don't need to update nodes for the last chi angle since there is no next chi angle to predict.
            if chi_idx == 3:
                break
            
            # Update protein node embeddings with predicted chi angles.
            gvp_layer, gvp_norm = self.chi_vector_update_layers[chi_idx], self.chi_vector_layer_norms[chi_idx]
            concat_features = EquivariantData(torch.cat([prot_scalars, prev_chi], dim=1), prot_nodes.vectors)
            prot_nodes = gvp_norm(gvp_layer(concat_features))
            prot_scalars = prot_nodes.scalars

        return (
            sequence_logits, output_chi_logits, encoder_nodes, encoder_eattrs, lig_nodes.scalars.detach().clone(), lig_pr_eattr.detach().clone(), decoding_order_log_probs
        )

    def forward(
            self, batch: BatchData, return_nodes: bool = False, return_unconditional_probabilities: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main teacher-forced forward pass for training the model in a supervised manner.
        Args:
            return_unconditional_probabilities: to return the probabilities of all residues being the first to be decoded.
            return_full_conditional_probabilities: to return the probabilities of all residues being decoded before the current residue.
        """
        # Input sanity checks.
        assert batch.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."
        assert batch.decoding_order is not None, "Decoding order must be specified in batch data."

        # Apply encoding layers to build up protein and ligand representations.
        lig_nodes, prot_nodes, pr_pr_eattr, lig_pr_eattr = self.apply_encoding_layers(batch)

        # Create protein-protein edge mask, if True then source node was decoded before target node. Self-edges are False.
        ### Argsort on decoding order maps from node index to decoding order, so we can use this to compare decoding order of source and target nodes.
        decoding_order_sort_indices = batch.decoding_order.argsort()

        if return_unconditional_probabilities:
            # Provide nothing for all residues (mimics treating all residues as the first to be decoded).
            pr_pr_edge_mask = torch.zeros_like(batch.pr_pr_edge_index[0], dtype=torch.bool)
        else:
            # Otherwise, provide only residues that would be decoded before the current residue if we were autoregressively sampling.
            pr_pr_edge_mask = (decoding_order_sort_indices[batch.pr_pr_edge_index[0]] < decoding_order_sort_indices[batch.pr_pr_edge_index[1]])

        unmasked_edge_indices = batch.pr_pr_edge_index[:, pr_pr_edge_mask]
        unmasked_edge_attrs = pr_pr_eattr[pr_pr_edge_mask]
        masked_edge_indices = batch.pr_pr_edge_index[:, ~pr_pr_edge_mask]
        masked_edge_attrs = pr_pr_eattr[~pr_pr_edge_mask]

        # Use ground-truth chi angles discretized into bins with an associated continuous bin offset which we can learn jointly.
        chi_angle_encoding = self.rotamer_builder.compute_binned_degree_basis_function(batch.chi_angles).nan_to_num()
        chi_bin_center = chi_angle_encoding.argmax(dim=-1)
        discrete_chi_angle_encoding = F.one_hot(chi_bin_center, num_classes=self.rotamer_builder.num_chi_bins).float()
        chi_angle_discrete_degrees = self.rotamer_builder.index_to_degree_bin[chi_bin_center]
        true_chi_offset_pos = torch.remainder(batch.chi_angles - chi_angle_discrete_degrees, 360)
        true_chi_offset_neg = -torch.remainder(chi_angle_discrete_degrees - batch.chi_angles, 360)
        abs_min_indices = torch.stack([true_chi_offset_pos.abs(), true_chi_offset_neg.abs()], dim=-1).argmin(dim=-1)
        target_chi_offsets = torch.where(abs_min_indices == 0, true_chi_offset_pos, true_chi_offset_neg)

        binned_degree_chi_for_teacher_forcing = chi_angle_encoding.flatten(start_dim=1)[unmasked_edge_indices[0]]
        sequence_embedding_nodes = self.sequence_label_embedding(batch.sequence_indices)
        sequence_embedding_for_teacher_forcing = sequence_embedding_nodes[unmasked_edge_indices[0]]

        # Use decoding order mask to selectively provide label data for nodes that have been previously decoded.
        source_node_edge_features_exp_unmasked = torch.cat([sequence_embedding_for_teacher_forcing, binned_degree_chi_for_teacher_forcing, unmasked_edge_attrs], dim=1)

        # Fill sequence representations with 22nd mask token for nodes not yet decoded.
        masked_sequence_embeddings = self.sequence_label_embedding(torch.full_like(batch.sequence_indices, fill_value=21))
        masked_chi_encoding = torch.zeros(masked_edge_attrs.shape[0], binned_degree_chi_for_teacher_forcing.shape[1], device=batch.device)

        # Append edge features and encoder node features to nodes that have not yet been decoded, dont add protein node embeddings to unmasked nodes yet as these will be updated in decoding process.
        source_node_edge_features_exp_masked = torch.cat([masked_sequence_embeddings[masked_edge_indices[0]], masked_chi_encoding, masked_edge_attrs, prot_nodes.scalars[masked_edge_indices[0]]], dim=1)

        # Iteratively update protein nodes with teacher-forced decoding.
        for didx, decoder_layer in enumerate(self.protein_decoder_layers):
            # Provides node features for unmasked source nodes at current step of decoding.
            curr_nodes_exp = prot_nodes.scalars[unmasked_edge_indices[0]]
            curr_edge_features_unmasked = torch.cat([source_node_edge_features_exp_unmasked, curr_nodes_exp], dim=1)

            # Add self-edges to nodes that have not yet been decoded.
            all_edge_features = torch.cat([source_node_edge_features_exp_masked, curr_edge_features_unmasked], dim=0)
            all_edge_indices = torch.cat([masked_edge_indices, unmasked_edge_indices], dim=1)

            # Update node embeddings with teacher-forced edges.
            prot_nodes = decoder_layer(all_edge_features, prot_nodes, all_edge_indices, lig_nodes, lig_pr_eattr, batch.lig_pr_edge_index)

        # Compute sequence logits from final protein node embeddings.
        sequence_logits = self.sequence_output_layer(prot_nodes.scalars)

        # Teacher-force chi angle prediction from final protein node embeddings.
        prev_chi = torch.empty((batch.num_residues, 0), device=self.device)
        output_sampled_angles = torch.full((batch.num_residues, 4), fill_value=torch.nan, device=self.device)
        sampled_angle_mask = ~batch.chi_angles.isnan()

        prot_scalars = prot_nodes.scalars
        output_chi_logits = torch.full((batch.num_residues, 4, self.rotamer_builder.num_chi_bins), fill_value=torch.nan, device=self.device, dtype=prot_scalars.dtype)
        output_chi_offsets = torch.full((batch.num_residues, 4), fill_value=torch.nan, device=self.device, dtype=prot_scalars.dtype)
        for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
            # Predict a chi angle from final protein node embeddings.
            chi_logits = chi_layer(torch.cat([prot_scalars, sequence_embedding_nodes, prev_chi], dim=1))
            curr_masks = sampled_angle_mask[:, chi_idx]

            # Predict the offset from the discretized chi angle bin means. Combine the discrete bin repr. with logits to pred the offset.
            pred_chi_offset = self.chi_offset_prediction_layers[chi_idx](torch.cat([chi_logits, discrete_chi_angle_encoding[:, chi_idx]], dim=1)).squeeze()

            output_chi_logits[curr_masks, chi_idx] = chi_logits[curr_masks]
            output_chi_offsets[curr_masks, chi_idx] = pred_chi_offset[curr_masks]

            # Teacher force with encoded ground-truth chi angles for next step of decoding.
            prev_chi = chi_angle_encoding[:, :chi_idx + 1].flatten(start_dim=1)

            # Don't need to update nodes for the last chi angle since there is no next chi angle to predict.
            if chi_idx == 3:
                break
            
            # Update protein node embeddings with predicted chi angles.
            gvp_layer, gvp_norm = self.chi_vector_update_layers[chi_idx], self.chi_vector_layer_norms[chi_idx]
            concat_features = EquivariantData(torch.cat([prot_scalars, prev_chi], dim=1), prot_nodes.vectors)
            prot_nodes = gvp_norm(gvp_layer(concat_features))
            prot_scalars = prot_nodes.scalars

        if not return_nodes:
            return sequence_logits, output_sampled_angles, output_chi_logits, output_chi_offsets, target_chi_offsets
        else:
            return sequence_logits, output_chi_logits, prot_nodes, lig_nodes # type: ignore

    @torch.no_grad()
    def tied_sample(
            self, batch1: BatchData, batch2: BatchData, lambda_: float = 0.5, sequence_sample_temperature: Optional[Union[float, torch.Tensor]] = None, 
            budget_residue_mask: Optional[torch.Tensor] = None, ala_budget: int = 4, gly_budget: int = 0,
            chi_angle_sample_temperature: Optional[float] = None, disabled_residues: Optional[list] = ['X'], 
            disable_pbar: bool = False, repack_all: bool = False
    ) -> Tuple[Sampled_Output, Sampled_Output]:
        """
        lambda_: is the mixing coefficient for the probabilities: (lambda_ * P1) + ((1 - lambda_) * P2)
        """
        # Input sanity checks.
        assert batch1.pr_pr_edge_index is not None and batch2.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch1.pr_pr_edge_distance is not None and batch2.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."
        assert batch1.lig_pr_edge_index is not None and batch2.lig_pr_edge_index is not None, "Protein-ligand edge index must be specified in batch data."
        assert batch1.lig_pr_edge_distance is not None and batch2.lig_pr_edge_distance is not None, "Protein-ligand edge distance must be specified in batch data."
        assert batch1.decoding_order is not None and batch2.decoding_order is not None, "Decoding order must be specified in batch data."
        assert lambda_ >= 0.0 and lambda_ <= 1.0, "Lambda must be between 0 and 1."

        assert batch1.num_residues == batch2.num_residues, "Batch1 and Batch2 must have the same number of residues."
        if (batch1.chain_mask != batch2.chain_mask).all().item():
            print("Warning! Chain masks are not the same, using the chain mask and taking rotamers from batch1.")

        if (batch1.sequence_indices != batch2.sequence_indices).all().item():
            print("Warning! Sequence indices are not the same, using the sequence indices from batch1.")

        if sequence_sample_temperature is not None:
            assert (isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.shape[0] == batch.num_residues or sequence_sample_temperature.numel() == 1)) or isinstance(sequence_sample_temperature, (int, float)), f"Sequence sample temperature must be a scalar or a tensor of shape (num_residues,). Got {sequence_sample_temperature}."

        if budget_residue_mask is None:
            budget_residue_mask = torch.zeros(batch1.num_residues, dtype=torch.bool, device=self.device)

        # 1 in chain mask tells us to sample sequence, a 0 tells us to use the input sequence stored in batch.sequence_indices.
        input_chi_angles_1 = batch1.chi_angles.nan_to_num()
        input_chi_angles_2 = batch2.chi_angles.nan_to_num()
        chain_mask = batch1.chain_mask.long()

        # Apply encoding layers to build up protein and ligand representations.
        lig_nodes1, prot_nodes1, pr_pr_eattr1, lig_pr_eattr1 = self.apply_encoding_layers(batch1)
        lig_nodes2, prot_nodes2, pr_pr_eattr2, lig_pr_eattr2 = self.apply_encoding_layers(batch2)

        # Initialize sequence embeddings to 'NotDecoded' nodes.
        sequence_embeddings = self.sequence_label_embedding(torch.full_like(batch1.sequence_indices, fill_value=21))
        # Expanded source encoder node + edge features for source nodes that have not yet been decoded.
        masked_sequence_embedding1 = sequence_embeddings.clone()[batch1.pr_pr_edge_index[0]]
        masked_sequence_embedding2 = sequence_embeddings.clone()[batch2.pr_pr_edge_index[0]]
        masked_chi_embedding1 = torch.zeros((pr_pr_eattr1.shape[0], self.chi_embedding_dim,), device=self.device)
        masked_chi_embedding2 = torch.zeros((pr_pr_eattr2.shape[0], self.chi_embedding_dim,), device=self.device)

        source_node_edge_features_exp_masked_1 = torch.cat([masked_sequence_embedding1, masked_chi_embedding1, pr_pr_eattr1, prot_nodes1.scalars[batch1.pr_pr_edge_index[0]]], dim=1)
        source_node_edge_features_exp_masked_2 = torch.cat([masked_sequence_embedding2, masked_chi_embedding2, pr_pr_eattr2, prot_nodes2.scalars[batch2.pr_pr_edge_index[0]]], dim=1)

        # The decoding orders should be the same so we can just keep track of one batch's.
        # Drops the NaN padding from the decoding order and flattens the batch dimension so 
        #   we can get sort indices of the same shape as number of nodes
        decoding_order_sort_indices = batch1.decoding_order.argsort(dim=-1)[~batch1.decoding_order.isnan()]

        # Initialize placeholder output tensors.
        prot_node_stack_1 = [prot_nodes1] + [prot_nodes1.get_zeroslike() for _ in range(self.num_decoder_layers)]
        prot_node_stack_2 = [prot_nodes2] + [prot_nodes2.get_zeroslike() for _ in range(self.num_decoder_layers)]
        output_tensors_1 = create_sampling_output(batch1.num_residues, self.rotamer_builder.num_chi_bins, self.device)
        output_tensors_2 = create_sampling_output(batch2.num_residues, self.rotamer_builder.num_chi_bins, self.device)

        # Iteratively decode protein nodes.
        for idx in tqdm(range(batch1.decoding_order.shape[1]), total=batch1.decoding_order.shape[1], leave=False, dynamic_ncols=True, desc='Running model.sample()', disable=disable_pbar):
            # Select current row in the batched decoding order.
            node_idces = batch1.decoding_order[:, idx]
            node_idces = node_idces[~node_idces.isnan()].long()

            per_batch_ala_counts = scatter((output_tensors_1.sampled_sequence_indices == aa_short_to_idx['A']).float(), batch1.batch_indices, reduce='sum', dim_size=len(batch1.batch_indices.unique()))
            per_batch_gly_counts = scatter((output_tensors_1.sampled_sequence_indices == aa_short_to_idx['G']).float(), batch1.batch_indices, reduce='sum', dim_size=len(batch1.batch_indices.unique()))
            per_res_ala_over_budget = (per_batch_ala_counts >= ala_budget)[batch1.batch_indices]
            per_res_gly_over_budget = (per_batch_gly_counts >= gly_budget)[batch1.batch_indices]
            curr_res_ala_over_budget = per_res_ala_over_budget[node_idces] & budget_residue_mask[node_idces]
            curr_res_gly_over_budget = per_res_gly_over_budget[node_idces] & budget_residue_mask[node_idces]

            # 1 in chain mask tells us to take sequence/chi from input data, a 0 tells us to sample it with the model.
            curr_chain_mask = chain_mask[node_idces]

            # Get just the edges that are incident to the current node.
            edges_sink_curr_nodes_mask_1 = torch.isin(batch1.pr_pr_edge_index[1], node_idces)
            sink_curr_edge_indices_1 = batch1.pr_pr_edge_index[:, edges_sink_curr_nodes_mask_1]
            sink_curr_edge_features_1 = pr_pr_eattr1[edges_sink_curr_nodes_mask_1, :]

            edges_sink_curr_nodes_mask_2 = torch.isin(batch2.pr_pr_edge_index[1], node_idces)
            sink_curr_edge_indices_2 = batch2.pr_pr_edge_index[:, edges_sink_curr_nodes_mask_2]
            sink_curr_edge_features_2 = pr_pr_eattr2[edges_sink_curr_nodes_mask_2, :]

            # True if source node was decoded before target node, False otherwise.
            pr_pr_edge_mask_1 = (decoding_order_sort_indices[sink_curr_edge_indices_1[0]] < decoding_order_sort_indices[sink_curr_edge_indices_1[1]])
            pr_pr_edge_mask_2 = (decoding_order_sort_indices[sink_curr_edge_indices_2[0]] < decoding_order_sort_indices[sink_curr_edge_indices_2[1]])

            unmasked_edge_indices_1 = sink_curr_edge_indices_1[:, pr_pr_edge_mask_1]
            unmasked_edge_attrs_1 = sink_curr_edge_features_1[pr_pr_edge_mask_1]
            masked_edge_indices_1 = sink_curr_edge_indices_1[:, ~pr_pr_edge_mask_1]
            masked_self_edges_1 = source_node_edge_features_exp_masked_1[edges_sink_curr_nodes_mask_1][~pr_pr_edge_mask_1]

            unmasked_edge_indices_2 = sink_curr_edge_indices_2[:, pr_pr_edge_mask_2]
            unmasked_edge_attrs_2 = sink_curr_edge_features_2[pr_pr_edge_mask_2]
            masked_edge_indices_2 = sink_curr_edge_indices_2[:, ~pr_pr_edge_mask_2]
            masked_self_edges_2 = source_node_edge_features_exp_masked_2[edges_sink_curr_nodes_mask_2][~pr_pr_edge_mask_2]

            lig_prot_edges_sink_curr_nodes_mask_1 = torch.isin(batch1.lig_pr_edge_index[1], node_idces)
            curr_lig_prot_eidx_1 = batch1.lig_pr_edge_index[:, lig_prot_edges_sink_curr_nodes_mask_1]
            curr_lig_prot_eattr_1 = lig_pr_eattr1[lig_prot_edges_sink_curr_nodes_mask_1, :]

            lig_prot_edges_sink_curr_nodes_mask_2 = torch.isin(batch2.lig_pr_edge_index[1], node_idces)
            curr_lig_prot_eidx_2 = batch2.lig_pr_edge_index[:, lig_prot_edges_sink_curr_nodes_mask_2]
            curr_lig_prot_eattr_2 = lig_pr_eattr2[lig_prot_edges_sink_curr_nodes_mask_2, :]

            # Extract previously decoded source node features.
            source_prev_sampled_labels_1 = sequence_embeddings[unmasked_edge_indices_1[0]]
            source_prev_chi_angles_1 = output_tensors_1.sampled_chi_encoding.nan_to_num().flatten(start_dim=1)[unmasked_edge_indices_1[0]]

            source_prev_sampled_labels_2 = sequence_embeddings[unmasked_edge_indices_2[0]]
            source_prev_chi_angles_2 = output_tensors_2.sampled_chi_encoding.nan_to_num().flatten(start_dim=1)[unmasked_edge_indices_2[0]]

            curr_sink_node_edge_features_unmasked_1 = torch.cat([source_prev_sampled_labels_1, source_prev_chi_angles_1, unmasked_edge_attrs_1], dim=1)
            curr_sink_node_edge_features_unmasked_2 = torch.cat([source_prev_sampled_labels_2, source_prev_chi_angles_2, unmasked_edge_attrs_2], dim=1)
            for layer_idx, decoder_layer in enumerate(self.protein_decoder_layers):
                # Provides node features for unmasked source nodes at current step of decoding.
                curr_idx_nodes_1 = prot_node_stack_1[layer_idx]
                curr_idx_nodes_2 = prot_node_stack_2[layer_idx]

                curr_idx_node_scalars_exp_1 = curr_idx_nodes_1.scalars[unmasked_edge_indices_1[0]]
                curr_idx_node_scalars_exp_2 = curr_idx_nodes_2.scalars[unmasked_edge_indices_2[0]]

                curr_decoded_edge_features_1 = torch.cat([curr_sink_node_edge_features_unmasked_1, curr_idx_node_scalars_exp_1], dim=1)
                curr_decoded_edge_features_2 = torch.cat([curr_sink_node_edge_features_unmasked_2, curr_idx_node_scalars_exp_2], dim=1)

                # Add self-edges for nodes that have not yet been decoded.
                curr_decoded_edge_features_1 = torch.cat([masked_self_edges_1, curr_decoded_edge_features_1], dim=0)
                curr_decoded_edge_features_2 = torch.cat([masked_self_edges_2, curr_decoded_edge_features_2], dim=0)

                curr_decoded_edge_indices_1 = torch.cat([masked_edge_indices_1, unmasked_edge_indices_1], dim=1)
                curr_decoded_edge_indices_2 = torch.cat([masked_edge_indices_2, unmasked_edge_indices_2], dim=1)

                # Update node embeddings for current step of decoding using edges terminating at current sink nodes.
                updated_curr_nodes_1 = decoder_layer(curr_decoded_edge_features_1, prot_node_stack_1[layer_idx], curr_decoded_edge_indices_1, lig_nodes1, curr_lig_prot_eattr_1, curr_lig_prot_eidx_1).get_indices(node_idces)
                updated_curr_nodes_2 = decoder_layer(curr_decoded_edge_features_2, prot_node_stack_2[layer_idx], curr_decoded_edge_indices_2, lig_nodes2, curr_lig_prot_eattr_2, curr_lig_prot_eidx_2).get_indices(node_idces)

                prot_node_stack_1[layer_idx + 1].set_indices(node_idces, updated_curr_nodes_1)
                prot_node_stack_2[layer_idx + 1].set_indices(node_idces, updated_curr_nodes_2)
            
            # Convert node embeddings to logits for sequence prediction.
            curr_out_logits_1 = self.sequence_output_layer(prot_node_stack_1[-1].scalars[node_idces])
            curr_out_logits_2 = self.sequence_output_layer(prot_node_stack_2[-1].scalars[node_idces])
            if disabled_residues is not None:
                sampling_residue_mask = ~(curr_chain_mask.bool())
                for res_short in disabled_residues:
                    curr_out_logits_1[sampling_residue_mask, aa_short_to_idx[res_short]] = torch.finfo(curr_out_logits_1.dtype).min
                    curr_out_logits_2[sampling_residue_mask, aa_short_to_idx[res_short]] = torch.finfo(curr_out_logits_2.dtype).min
                
            curr_out_logits_1[curr_res_ala_over_budget, aa_short_to_idx['A']] = torch.finfo(curr_out_logits_1.dtype).min
            curr_out_logits_1[curr_res_gly_over_budget, aa_short_to_idx['G']] = torch.finfo(curr_out_logits_1.dtype).min
            curr_out_logits_2[curr_res_ala_over_budget, aa_short_to_idx['A']] = torch.finfo(curr_out_logits_2.dtype).min
            curr_out_logits_2[curr_res_gly_over_budget, aa_short_to_idx['G']] = torch.finfo(curr_out_logits_2.dtype).min

            # Sample sequence indices if temperature is specified, otherwise take argmax.
            if sequence_sample_temperature is None:
                sequence_sample_temperature = 1e-6
            # If sequence_sample_temperature is a scalar, broadcast it to the number of residues.
            if not isinstance(sequence_sample_temperature, torch.Tensor) or (sequence_sample_temperature.numel() == 1):
                sequence_sample_temperature = torch.ones(batch1.num_residues, device=self.device) * sequence_sample_temperature
            curr_out_probs_1 = torch.softmax(curr_out_logits_1 / sequence_sample_temperature[node_idces].unsqueeze(-1), dim=-1)
            curr_out_probs_2 = torch.softmax(curr_out_logits_2 / sequence_sample_temperature[node_idces].unsqueeze(-1), dim=-1)

            # HANDLE LAMBDA MIXING OF PROBABILITIES.  ( > 0.5 favors batch1, < 0.5 favors batch2)
            interpolated_probs = (lambda_ * curr_out_probs_1) + ((1 - lambda_) * curr_out_probs_2)
            curr_out_sample = torch.distributions.Categorical(probs=interpolated_probs).sample()

            # Use chain_mask to select from input sequence for partial-sequence design as needed.
            sampled_or_fixed_sequence_idx = (curr_chain_mask * batch1.sequence_indices[node_idces]) + ((1 - curr_chain_mask) * curr_out_sample)

            # Update sequence embeddings with sampled or fixed sequence indices.
            sequence_embeddings[node_idces] = self.sequence_label_embedding(sampled_or_fixed_sequence_idx)

            # Create masks for whether each chi angle is defined for each residue being decoded.
            #   Handle X residues by converting to Gly and sanity check we didn't sample them unless provided by chain_mask.
            sampled_x_residue_mask = torch.full_like(sampled_or_fixed_sequence_idx, aa_short_to_idx['X']) == sampled_or_fixed_sequence_idx
            if sampled_x_residue_mask.any().item() and (~batch1.chain_mask[node_idces][sampled_x_residue_mask]).any().item():
                # Allow X residues to be sampled if not disabling them or chain_mask is 0, otherwise crashes with assertion error.
                assert (disabled_residues is None) or (not 'X' in disabled_residues), "Sampled an X residue when sampling X residues is disabled."
            x_to_gly_sampled_or_fixed_sequence_idx = sampled_or_fixed_sequence_idx.clone()
            x_to_gly_sampled_or_fixed_sequence_idx[sampled_x_residue_mask] = aa_short_to_idx['G']
            sampled_sequence_chi_masks = self.rotamer_builder.aa_to_chi_angle_mask[x_to_gly_sampled_or_fixed_sequence_idx] # type: ignore

            # Store sequence logits and sampled sequence indices in output tensors.
            output_tensors_1.sequence_logits[node_idces] = curr_out_logits_1
            output_tensors_1.sampled_sequence_indices[node_idces] = sampled_or_fixed_sequence_idx

            output_tensors_2.sequence_logits[node_idces] = curr_out_logits_2
            output_tensors_2.sampled_sequence_indices[node_idces] = sampled_or_fixed_sequence_idx

            # Decode chi angles from final protein node embeddings.
            #   pull previously decoded chi angles from output tensors.
            curr_nodes_decoder_embeddings_1 = prot_node_stack_1[-1].get_indices(node_idces)
            curr_nodes_decoder_embeddings_2 = prot_node_stack_2[-1].get_indices(node_idces)

            prot_scalars_1 = curr_nodes_decoder_embeddings_1.scalars
            prot_scalars_2 = curr_nodes_decoder_embeddings_2.scalars

            chi_prev_1 = torch.empty((node_idces.shape[0], 0), device=self.device)
            chi_prev_2 = torch.empty((node_idces.shape[0], 0), device=self.device)

            for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
                # Predict chi angles from final protein node embeddings.
                chi_logits_1 = chi_layer(torch.cat([prot_scalars_1, sequence_embeddings[node_idces], chi_prev_1], dim=1))
                chi_logits_2 = chi_layer(torch.cat([prot_scalars_2, sequence_embeddings[node_idces], chi_prev_2], dim=1))

                # Sample chi angles if temperature is specified, otherwise take argmax.
                if chi_angle_sample_temperature is None:
                    chi_sample_1 = chi_logits_1.argmax(dim=-1)
                    chi_sample_2 = chi_logits_2.argmax(dim=-1)
                else:
                    chi_probs_1 = torch.softmax(chi_logits_1 / chi_angle_sample_temperature, dim=-1)
                    chi_probs_2 = torch.softmax(chi_logits_2 / chi_angle_sample_temperature, dim=-1)

                    chi_sample_1 = torch.distributions.Categorical(probs=chi_probs_1).sample()
                    chi_sample_2 = torch.distributions.Categorical(probs=chi_probs_2).sample()

                chi_sample_one_hot_1 = F.one_hot(chi_sample_1, num_classes=self.rotamer_builder.num_chi_bins).float()
                chi_sample_offset_1 = self.chi_offset_prediction_layers[chi_idx](torch.cat([chi_logits_1, chi_sample_one_hot_1], dim=1)).squeeze()

                chi_sample_one_hot_2 = F.one_hot(chi_sample_2, num_classes=self.rotamer_builder.num_chi_bins).float()
                chi_sample_offset_2 = self.chi_offset_prediction_layers[chi_idx](torch.cat([chi_logits_2, chi_sample_one_hot_2], dim=1)).squeeze()

                # Convert sampled index to angle, then to RBF encoding.
                #   1 in chain mask tells us to sample chi angle, a 0 tells us to use the input chi angle.
                sampled_angles_1 = torch.remainder(self.rotamer_builder.index_to_degree_bin[chi_sample_1] + chi_sample_offset_1 + 180, 360) - 180 # type: ignore
                sampled_angles_2 = torch.remainder(self.rotamer_builder.index_to_degree_bin[chi_sample_2] + chi_sample_offset_2 + 180, 360) - 180 # type: ignore

                if not repack_all:
                    sampled_angles_1 = (curr_chain_mask * input_chi_angles_1[node_idces, chi_idx]) + ((1 - curr_chain_mask) * sampled_angles_1)
                    sampled_angles_2 = (curr_chain_mask * input_chi_angles_2[node_idces, chi_idx]) + ((1 - curr_chain_mask) * sampled_angles_2)

                curr_chi_encoding_1 = self.rotamer_builder.compute_binned_degree_basis_function(sampled_angles_1.unsqueeze(-1)).squeeze(1)
                chi_prev_1 = torch.cat([chi_prev_1, curr_chi_encoding_1], dim=1)

                curr_chi_encoding_2 = self.rotamer_builder.compute_binned_degree_basis_function(sampled_angles_2.unsqueeze(-1)).squeeze(1)
                chi_prev_2 = torch.cat([chi_prev_2, curr_chi_encoding_2], dim=1)

                # Convert chi angle encoding to logits and store in output tensors.
                curr_chi_masks = sampled_sequence_chi_masks[:, chi_idx]
                output_tensors_1.chi_logits[node_idces[curr_chi_masks], chi_idx] = chi_logits_1[curr_chi_masks]
                output_tensors_1.sampled_chi_encoding[node_idces[curr_chi_masks], chi_idx] = curr_chi_encoding_1[curr_chi_masks]
                output_tensors_1.sampled_chi_degrees[node_idces[curr_chi_masks], chi_idx] = sampled_angles_1[curr_chi_masks]

                output_tensors_2.chi_logits[node_idces[curr_chi_masks], chi_idx] = chi_logits_2[curr_chi_masks]
                output_tensors_2.sampled_chi_encoding[node_idces[curr_chi_masks], chi_idx] = curr_chi_encoding_2[curr_chi_masks]
                output_tensors_2.sampled_chi_degrees[node_idces[curr_chi_masks], chi_idx] = sampled_angles_2[curr_chi_masks]

                if chi_idx == 3:
                    # Don't need to update node representations for the last chi angle since there is no next chi angle to predict.
                    break

                gvp_layer, gvp_norm = self.chi_vector_update_layers[chi_idx], self.chi_vector_layer_norms[chi_idx]

                concat_features_1 = EquivariantData(torch.cat([prot_scalars_1, chi_prev_1], dim=1), curr_nodes_decoder_embeddings_1.vectors)
                curr_nodes_decoder_embeddings_1 = gvp_norm(gvp_layer(concat_features_1))
                prot_scalars_1 = curr_nodes_decoder_embeddings_1.scalars

                concat_features_2 = EquivariantData(torch.cat([prot_scalars_2, chi_prev_2], dim=1), curr_nodes_decoder_embeddings_2.vectors)
                curr_nodes_decoder_embeddings_2 = gvp_norm(gvp_layer(concat_features_2))
                prot_scalars_2 = curr_nodes_decoder_embeddings_2.scalars
        
        return output_tensors_1, output_tensors_2
    
    @torch.no_grad()
    def sample(
            self, batch: BatchData, 
            budget_residue_mask: Optional[torch.Tensor] = None, ala_budget: int = 4, gly_budget: int = 0,
            sequence_sample_temperature: Optional[Union[float, torch.Tensor]] = None, 
            chi_angle_sample_temperature: Optional[float] = None, disabled_residues: Optional[list] = ['X'], 
            disable_pbar: bool = False, return_encoder_embeddings: bool = False, chi_min_p: float = 0.0, seq_min_p: float = 0.0,
            ignore_chain_mask_zeros: bool = False, disable_charged_residue_mask: Optional[torch.Tensor] = None, repack_all: bool = False,
    ) -> Sampled_Output:
        """
        If ignore_chain_mask_zeros is true, than ONLY sample residues that are True/1 in the chain mask and return XAA residues for all unsampled residues.
        otherwise, samples all residues EXCEPT those that are True/1 in the chain mask as it will take the input sequence/rotamer for these residues.
        disable_charged
        """
        # Input sanity checks.
        assert batch.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."
        assert batch.lig_pr_edge_index is not None, "Protein-ligand edge index must be specified in batch data."
        assert batch.lig_pr_edge_distance is not None, "Protein-ligand edge distance must be specified in batch data."
        assert batch.decoding_order is not None, "Decoding order must be specified in batch data."
        if sequence_sample_temperature is not None:
            assert (isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.shape[0] == batch.num_residues or sequence_sample_temperature.numel() == 1)) or isinstance(sequence_sample_temperature, (int, float)), f"Sequence sample temperature must be a scalar or a tensor of shape (num_residues,). Got {sequence_sample_temperature}."

        if budget_residue_mask is None:
            budget_residue_mask = torch.zeros(batch.num_residues, dtype=torch.bool, device=self.device)

        input_chi_angles = batch.chi_angles.clone()

        # 1 in chain mask tells us to sample sequence, a 0 tells us to use the input sequence stored in batch.sequence_indices.
        chain_mask = batch.chain_mask.long()

        # Apply encoding layers to build up protein and ligand representations.
        lig_nodes, prot_nodes, pr_pr_eattr, lig_pr_eattr = self.apply_encoding_layers(batch)

        # Initialize sequence embeddings to 'NotDecoded' nodes.
        sequence_embeddings = self.sequence_label_embedding(torch.full_like(batch.sequence_indices, fill_value=21))

        # Expanded source encoder node + edge features for source nodes that have not yet been decoded.
        masked_sequence_embedding = sequence_embeddings.clone()[batch.pr_pr_edge_index[0]]
        masked_chi_embeding = torch.zeros((pr_pr_eattr.shape[0], self.chi_embedding_dim,), device=self.device)
        source_node_edge_features_exp_masked = torch.cat([masked_sequence_embedding, masked_chi_embeding, pr_pr_eattr, prot_nodes.scalars[batch.pr_pr_edge_index[0]]], dim=1)

        # Drops the NaN padding from the decoding order and flattens the batch dimension so 
        #   we can get sort indices of the same shape as number of nodes
        decoding_order_sort_indices = batch.decoding_order.argsort(dim=-1)[~batch.decoding_order.isnan()]

        # Initialize output tensors.
        prot_node_stack = [prot_nodes] + [prot_nodes.get_zeroslike() for _ in range(self.num_decoder_layers)]
        output_tensors = create_sampling_output(batch.num_residues, self.rotamer_builder.num_chi_bins, self.device)

        # Iteratively decode protein nodes.
        for idx in tqdm(range(batch.decoding_order.shape[1]), total=batch.decoding_order.shape[1], leave=False, dynamic_ncols=True, desc='Running model.sample()', disable=disable_pbar):
            # Select current row in the batched decoding order.
            node_idces = batch.decoding_order[:, idx]
            node_idces = node_idces[~node_idces.isnan()].long()

            per_batch_ala_counts = scatter((output_tensors.sampled_sequence_indices == aa_short_to_idx['A']).float(), batch.batch_indices, reduce='sum', dim_size=len(batch.batch_indices.unique()))
            per_res_ala_over_budget = (per_batch_ala_counts >= ala_budget)[batch.batch_indices]
            curr_res_ala_over_budget = per_res_ala_over_budget[node_idces] & budget_residue_mask[node_idces]

            per_batch_gly_counts = scatter((output_tensors.sampled_sequence_indices == aa_short_to_idx['G']).float(), batch.batch_indices, reduce='sum', dim_size=len(batch.batch_indices.unique()))
            per_res_gly_over_budget = (per_batch_gly_counts >= gly_budget)[batch.batch_indices]
            curr_res_gly_over_budget = per_res_gly_over_budget[node_idces] & budget_residue_mask[node_idces]

            # 1 in chain mask tells us to take sequence/chi from input data, a 0 tells us to sample it with the model.
            curr_chain_mask = chain_mask[node_idces]

            if ignore_chain_mask_zeros:
                node_idces = node_idces[curr_chain_mask.bool()]
                if node_idces.numel() == 0:
                    continue

            # Get just the edges that are incident to the current node.
            edges_sink_curr_nodes_mask = torch.isin(batch.pr_pr_edge_index[1], node_idces)
            sink_curr_edge_indices = batch.pr_pr_edge_index[:, edges_sink_curr_nodes_mask]
            sink_curr_edge_features = pr_pr_eattr[edges_sink_curr_nodes_mask, :]

            # True if source node was decoded before target node, False otherwise.
            pr_pr_edge_mask = (decoding_order_sort_indices[sink_curr_edge_indices[0]] < decoding_order_sort_indices[sink_curr_edge_indices[1]])
            unmasked_edge_indices = sink_curr_edge_indices[:, pr_pr_edge_mask]
            unmasked_edge_attrs = sink_curr_edge_features[pr_pr_edge_mask]
            masked_edge_indices = sink_curr_edge_indices[:, ~pr_pr_edge_mask]
            masked_self_edges = source_node_edge_features_exp_masked[edges_sink_curr_nodes_mask][~pr_pr_edge_mask]

            lig_prot_edges_sink_curr_nodes_mask = torch.isin(batch.lig_pr_edge_index[1], node_idces)
            curr_lig_prot_eidx = batch.lig_pr_edge_index[:, lig_prot_edges_sink_curr_nodes_mask]
            curr_lig_prot_eattr = lig_pr_eattr[lig_prot_edges_sink_curr_nodes_mask, :]

            # Extract previously decoded source node features.
            source_prev_sampled_labels = sequence_embeddings[unmasked_edge_indices[0]]
            source_prev_chi_angles = output_tensors.sampled_chi_encoding.nan_to_num().flatten(start_dim=1)[unmasked_edge_indices[0]]

            curr_sink_node_edge_features_unmasked = torch.cat([source_prev_sampled_labels, source_prev_chi_angles, unmasked_edge_attrs], dim=1)
            for layer_idx, decoder_layer in enumerate(self.protein_decoder_layers):
                # Provides node features for unmasked source nodes at current step of decoding.
                curr_idx_nodes = prot_node_stack[layer_idx]
                curr_idx_node_scalars_exp = curr_idx_nodes.scalars[unmasked_edge_indices[0]]
                curr_decoded_edge_features = torch.cat([curr_sink_node_edge_features_unmasked, curr_idx_node_scalars_exp], dim=1)

                # Add self-edges for nodes that have not yet been decoded.
                curr_decoded_edge_features = torch.cat([masked_self_edges, curr_decoded_edge_features], dim=0)
                curr_decoded_edge_indices = torch.cat([masked_edge_indices, unmasked_edge_indices], dim=1)

                # Update node embeddings for current step of decoding using edges terminating at current sink nodes.
                updated_curr_nodes = decoder_layer(curr_decoded_edge_features, prot_node_stack[layer_idx], curr_decoded_edge_indices, lig_nodes, curr_lig_prot_eattr, curr_lig_prot_eidx).get_indices(node_idces)
                prot_node_stack[layer_idx + 1].set_indices(node_idces, updated_curr_nodes)

            # Convert node embeddings to logits for sequence prediction.
            curr_out_logits = self.sequence_output_layer(prot_node_stack[-1].scalars[node_idces])
            if disabled_residues is not None:
                sampling_residue_mask = ~(curr_chain_mask.bool()) if not ignore_chain_mask_zeros else curr_chain_mask.bool()
                for res_short in disabled_residues:
                    curr_out_logits[sampling_residue_mask, aa_short_to_idx[res_short]] = torch.finfo(curr_out_logits.dtype).min

            curr_out_logits[curr_res_ala_over_budget, aa_short_to_idx['A']] = torch.finfo(curr_out_logits.dtype).min
            curr_out_logits[curr_res_gly_over_budget, aa_short_to_idx['G']] = torch.finfo(curr_out_logits.dtype).min
            
            if disable_charged_residue_mask is not None:
                curr_disable_charged_mask = disable_charged_residue_mask[node_idces]
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['K']] = float('-Inf')
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['R']] = float('-Inf')
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['D']] = float('-Inf')
                curr_out_logits[curr_disable_charged_mask, aa_short_to_idx['E']] = float('-Inf')

            # Sample sequence indices if temperature is specified, otherwise take argmax.
            if sequence_sample_temperature is None:
                curr_out_sample = curr_out_logits.argmax(dim=-1)
            else:
                curr_out_logits = minp_warp_logits(curr_out_logits, seq_min_p)
                if isinstance(sequence_sample_temperature, torch.Tensor) and (sequence_sample_temperature.numel() > 1):
                    curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature[node_idces].unsqueeze(-1), dim=-1)
                else:
                    curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature, dim=-1)
                curr_out_sample = torch.distributions.Categorical(probs=curr_out_probs).sample()
            
            # Use chain_mask to select from input sequence for partial-sequence design as needed.
            if not ignore_chain_mask_zeros:
                sampled_or_fixed_sequence_idx = (curr_chain_mask * batch.sequence_indices[node_idces]) + ((1 - curr_chain_mask) * curr_out_sample)
            else:
                sampled_or_fixed_sequence_idx = curr_out_sample

            # Update sequence embeddings with sampled or fixed sequence indices.
            sequence_embeddings[node_idces] = self.sequence_label_embedding(sampled_or_fixed_sequence_idx)

            # Create masks for whether each chi angle is defined for each residue being decoded.
            #   Handle X residues by converting to Gly and sanity check we didn't sample them unless provided by chain_mask.
            sampled_x_residue_mask = torch.full_like(sampled_or_fixed_sequence_idx, aa_short_to_idx['X']) == sampled_or_fixed_sequence_idx
            if sampled_x_residue_mask.any().item() and (~batch.chain_mask[node_idces][sampled_x_residue_mask]).any().item():
                # Allow X residues to be sampled if not disabling them or chain_mask is 0, otherwise crashes with assertion error.
                assert (disabled_residues is None) or (not 'X' in disabled_residues), "Sampled an X residue when sampling X residues is disabled."
            x_to_gly_sampled_or_fixed_sequence_idx = sampled_or_fixed_sequence_idx.clone()
            x_to_gly_sampled_or_fixed_sequence_idx[sampled_x_residue_mask] = aa_short_to_idx['G']
            sampled_sequence_chi_masks = self.rotamer_builder.aa_to_chi_angle_mask[x_to_gly_sampled_or_fixed_sequence_idx] # type: ignore

            # Store sequence logits and sampled sequence indices in output tensors.
            output_tensors.sequence_logits[node_idces] = curr_out_logits
            output_tensors.sampled_sequence_indices[node_idces] = sampled_or_fixed_sequence_idx

            # Decode chi angles from final protein node embeddings.
            #   pull previously decoded chi angles from output tensors.
            curr_nodes_decoder_embeddings = prot_node_stack[-1].get_indices(node_idces)
            prot_scalars = curr_nodes_decoder_embeddings.scalars
            chi_prev = torch.empty((node_idces.shape[0], 0), device=self.device)
            for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
                # Predict chi angles from final protein node embeddings.
                chi_logits = chi_layer(torch.cat([prot_scalars, sequence_embeddings[node_idces], chi_prev], dim=1))

                # Sample chi angles if temperature is specified, otherwise take argmax.
                if chi_angle_sample_temperature is None:
                    chi_sample = chi_logits.argmax(dim=-1)
                else:
                    chi_logits = minp_warp_logits(chi_logits, chi_min_p)
                    chi_probs = torch.softmax(chi_logits / chi_angle_sample_temperature, dim=-1)
                    chi_sample = torch.distributions.Categorical(probs=chi_probs).sample()
                
                chi_sample_one_hot = F.one_hot(chi_sample, num_classes=self.rotamer_builder.num_chi_bins).float()
                chi_sample_offset = self.chi_offset_prediction_layers[chi_idx](torch.cat([chi_logits, chi_sample_one_hot], dim=1)).squeeze()

                # Convert sampled index to angle, then to RBF encoding.
                #   1 in chain mask tells us to sample chi angle, a 0 tells us to use the input chi angle.
                sampled_angles = torch.remainder(self.rotamer_builder.index_to_degree_bin[chi_sample] + chi_sample_offset + 180, 360) - 180 # type: ignore
                if not ignore_chain_mask_zeros and not repack_all:
                    nan_mask = ~input_chi_angles[node_idces, chi_idx].isnan()

                    # This is only 1 if we are both trying to fix the identity of the residue and the residue is not NaN
                    sample_mask = (curr_chain_mask.bool() & nan_mask).long()

                    # Keep sampled angles according to whether sample mask is 1 or 0.
                    sampled_angles = (sample_mask * input_chi_angles[node_idces, chi_idx].nan_to_num()) + ((1 - sample_mask) * sampled_angles)

                curr_chi_encoding = self.rotamer_builder.compute_binned_degree_basis_function(sampled_angles.unsqueeze(-1)).squeeze(1)
                chi_prev = torch.cat([chi_prev, curr_chi_encoding], dim=1)
                # curr_chi_encoding = torch.stack([torch.sin(sampled_angles.deg2rad()), torch.cos(sampled_angles.deg2rad())], dim=-1)
                # chi_prev = torch.cat([chi_prev, curr_chi_encoding], dim=1)

                # Convert chi angle encoding to logits and store in output tensors.
                curr_chi_masks = sampled_sequence_chi_masks[:, chi_idx]
                output_tensors.chi_logits[node_idces[curr_chi_masks], chi_idx] = chi_logits[curr_chi_masks]
                output_tensors.sampled_chi_encoding[node_idces[curr_chi_masks], chi_idx] = curr_chi_encoding[curr_chi_masks]
                output_tensors.sampled_chi_degrees[node_idces[curr_chi_masks], chi_idx] = sampled_angles[curr_chi_masks]

                if chi_idx == 3:
                    # Don't need to update node representations for the last chi angle since there is no next chi angle to predict.
                    break

                gvp_layer, gvp_norm = self.chi_vector_update_layers[chi_idx], self.chi_vector_layer_norms[chi_idx]
                concat_features = EquivariantData(torch.cat([prot_scalars, chi_prev], dim=1), curr_nodes_decoder_embeddings.vectors)
                curr_nodes_decoder_embeddings = gvp_norm(gvp_layer(concat_features))
                prot_scalars = curr_nodes_decoder_embeddings.scalars
        
        if ignore_chain_mask_zeros:
            output_tensors.sampled_sequence_indices[~batch.chain_mask] = aa_short_to_idx['X']
        
        if return_encoder_embeddings:
            return output_tensors, prot_nodes.scalars, pr_pr_eattr, lig_nodes.scalars, lig_pr_eattr # type: ignore
        return output_tensors

    def sample_by_lowest_entropy(
            self, batch: BatchData, sequence_sample_temperature: Optional[float] = None, 
            chi_angle_sample_temperature: Optional[float] = None, disabled_residues: Optional[list] = None, 
            disable_pbar: bool = False, return_encoder_embeddings: bool = False, fix_sequence: bool = False
    ):
        """
        Passes every un-decoded node through the decoder block and uses those embeddings to select one node to keep 
            for each subbatch with the decoding_order_degnerator module. This repeats until all nodes are decoded.
        """
        # Input sanity checks.
        assert batch.pr_pr_edge_index is not None, "Protein-protein edge index must be specified in batch data."
        assert batch.pr_pr_edge_distance is not None, "Protein-protein edge distance must be specified in batch data."
        assert batch.lig_pr_edge_index is not None, "Protein-ligand edge index must be specified in batch data."
        assert batch.lig_pr_edge_distance is not None, "Protein-ligand edge distance must be specified in batch data."

        input_chi_angles = batch.chi_angles.clone()

        # 1 in chain mask tells us to sample sequence, a 0 tells us to use the input sequence stored in batch.sequence_indices.
        chain_mask = batch.chain_mask.long()

        # Apply encoding layers to build up protein and ligand representations.
        lig_nodes, prot_nodes, pr_pr_eattr, lig_pr_eattr = self.apply_encoding_layers(batch)

        # Initialize sequence embeddings to 'NotDecoded' nodes.
        sequence_embeddings = self.sequence_label_embedding(torch.full_like(batch.sequence_indices, fill_value=21))

        # Expanded source encoder node + edge features for source nodes that have not yet been decoded.
        masked_sequence_embedding = sequence_embeddings.clone()[batch.pr_pr_edge_index[0]]
        masked_chi_embeding = torch.zeros((pr_pr_eattr.shape[0], self.chi_embedding_dim,), device=self.device)
        source_node_edge_features_exp_masked = torch.cat([masked_sequence_embedding, masked_chi_embeding, pr_pr_eattr, prot_nodes.scalars[batch.pr_pr_edge_index[0]]], dim=1)

        # Initialize output tensors.
        prot_node_stack = [prot_nodes] + [prot_nodes.get_zeroslike() for _ in range(self.num_decoder_layers)]
        output_tensors = create_sampling_output(batch.num_residues, self.rotamer_builder.num_chi_bins, self.device)

        # Decoding order generation tracking.
        output_decoding_order = []
        transition_probabilities = []
        sampled_transition_entropies = []
        node_decoded_mask = torch.zeros(batch.num_residues, dtype=torch.bool, device=self.device)

        # Iteratively decode protein nodes.
        _, max_num_per_batch_idx = batch.batch_indices.unique(return_counts=True)
        for _ in tqdm(range(max_num_per_batch_idx.max()), total=max_num_per_batch_idx.max().item(), leave=False, dynamic_ncols=True, desc='Running model.sample_learned_decoding_order()', disable=disable_pbar):

            # Remove nodes that have been decoded from the curr_node_idces
            curr_node_idces = (~node_decoded_mask).nonzero().flatten()

            # Get just the edges that are incident to the current node.
            edges_sink_curr_nodes_mask = torch.isin(batch.pr_pr_edge_index[1], curr_node_idces)
            sink_curr_edge_indices = batch.pr_pr_edge_index[:, edges_sink_curr_nodes_mask]
            sink_curr_edge_features = pr_pr_eattr[edges_sink_curr_nodes_mask, :]

            # 1 if source node was decoded before target node, 0 otherwise.
            pr_pr_edge_mask = node_decoded_mask[sink_curr_edge_indices[0]] # If True, source node was decoded before target node.
            unmasked_edge_indices = sink_curr_edge_indices[:, pr_pr_edge_mask]
            unmasked_edge_attrs = sink_curr_edge_features[pr_pr_edge_mask]
            masked_edge_indices = sink_curr_edge_indices[:, ~pr_pr_edge_mask]
            masked_self_edges = source_node_edge_features_exp_masked[edges_sink_curr_nodes_mask][~pr_pr_edge_mask]

            lig_prot_edges_sink_curr_nodes_mask = torch.isin(batch.lig_pr_edge_index[1], curr_node_idces)
            curr_lig_prot_eidx = batch.lig_pr_edge_index[:, lig_prot_edges_sink_curr_nodes_mask]
            curr_lig_prot_eattr = lig_pr_eattr[lig_prot_edges_sink_curr_nodes_mask, :]

            # Extract previously decoded source node features.
            source_prev_sampled_labels = sequence_embeddings[unmasked_edge_indices[0]]
            source_prev_chi_angles = output_tensors.sampled_chi_encoding.nan_to_num().flatten(start_dim=1)[unmasked_edge_indices[0]]

            # To implement learned decoding order, we need to destructively modify the prot_node_stack.
            node_stack_copy = [x.clone() for x in prot_node_stack]
            curr_sink_node_edge_features_unmasked = torch.cat([source_prev_sampled_labels, source_prev_chi_angles, unmasked_edge_attrs], dim=1)
            for layer_idx, decoder_layer in enumerate(self.protein_decoder_layers):
                # Get features for previously decoded nodes at the current stage of decoding.
                curr_idx_nodes = node_stack_copy[layer_idx]
                curr_idx_node_scalars_exp = curr_idx_nodes.scalars[unmasked_edge_indices[0]]
                curr_decoded_edge_features = torch.cat([curr_sink_node_edge_features_unmasked, curr_idx_node_scalars_exp], dim=1)

                # Add self-edges for with node features for those not yet been decoded.
                curr_decoded_edge_features = torch.cat([masked_self_edges, curr_decoded_edge_features], dim=0)
                curr_decoded_edge_indices = torch.cat([masked_edge_indices, unmasked_edge_indices], dim=1)

                # Update node embeddings for current step of decoding using edges terminating at current sink nodes.
                updated_curr_nodes = decoder_layer(curr_decoded_edge_features, node_stack_copy[layer_idx], curr_decoded_edge_indices, lig_nodes, curr_lig_prot_eattr, curr_lig_prot_eidx).get_indices(curr_node_idces)
                node_stack_copy[layer_idx + 1].set_indices(curr_node_idces, updated_curr_nodes)

            # Using the final node representations, select one to keep for each batch_idx.
            # Predict sequence for all nodes and take the lowest entropy node.
            sequence_logits = self.sequence_output_layer(updated_curr_nodes.scalars)
            sequence_entropy = torch.distributions.Categorical(logits=sequence_logits).entropy()

            curr_batch_idces = batch.batch_indices[curr_node_idces]
            _, decoded_indices = scatter_min(sequence_entropy, curr_batch_idces)
            decoded_indices = decoded_indices[curr_batch_idces.unique()]
            curr_idx_transition_probs = torch.ones_like(decoded_indices).float()
            sampled_transition_entropy = torch.tensor([0.0], device=self.device)

            # Convert back to global indices and update mask tracking decoding progress.
            curr_decoded_indices = curr_node_idces[decoded_indices] 
            transition_probabilities.append(curr_idx_transition_probs)
            sampled_transition_entropies.append(sampled_transition_entropy)
            output_decoding_order.append(curr_decoded_indices)

            # 1 in chain mask tells us to take sequence/chi from input data, a 0 tells us to sample it with the model.
            curr_chain_mask = chain_mask[curr_decoded_indices]
            node_decoded_mask[curr_decoded_indices] = True

            # Copy information for the selected nodes to the prot_node_stack.
            for layer_idx in range(len(prot_node_stack)):
                prot_node_stack[layer_idx].set_indices(curr_decoded_indices, node_stack_copy[layer_idx].get_indices(curr_decoded_indices))

            # Convert node embeddings to logits for sequence prediction.
            curr_out_logits = self.sequence_output_layer(prot_node_stack[-1].scalars[curr_decoded_indices])
            if disabled_residues is not None:
                for res_short in disabled_residues:
                    curr_out_logits[~(curr_chain_mask.bool()), aa_short_to_idx[res_short]] = torch.finfo(curr_out_logits.dtype).min
                
            # Sample sequence indices if temperature is specified, otherwise take argmax.
            if not fix_sequence:
                if sequence_sample_temperature is None:
                    curr_out_sample = curr_out_logits.argmax(dim=-1)
                else:
                    curr_out_probs = torch.softmax(curr_out_logits / sequence_sample_temperature, dim=-1)
                    curr_out_sample = torch.distributions.Categorical(probs=curr_out_probs).sample()
                # Use chain_mask to select from input sequence for partial-sequence design as needed.
                sampled_or_fixed_sequence_idx = (curr_chain_mask * batch.sequence_indices[curr_decoded_indices]) + ((1 - curr_chain_mask) * curr_out_sample)
            else:
                # Fix the sequence to the input structure sequence.
                sampled_or_fixed_sequence_idx = batch.sequence_indices[curr_decoded_indices]
            

            # Update sequence embeddings with sampled or fixed sequence indices.
            sequence_embeddings[curr_decoded_indices] = self.sequence_label_embedding(sampled_or_fixed_sequence_idx)

            # Create masks for whether each chi angle is defined for each residue being decoded.
            #   Handle X residues by converting to Gly and sanity check we didn't sample them unless provided by chain_mask.
            sampled_x_residue_mask = torch.full_like(sampled_or_fixed_sequence_idx, aa_short_to_idx['X']) == sampled_or_fixed_sequence_idx
            if sampled_x_residue_mask.any().item() and (~batch.chain_mask[curr_decoded_indices][sampled_x_residue_mask]).any().item():
                # Allow X residues to be sampled if not disabling them or chain_mask is 0, otherwise crashes with assertion error.
                assert (disabled_residues is None) or (not 'X' in disabled_residues), "Sampled an X residue when sampling X residues is disabled."
            x_to_gly_sampled_or_fixed_sequence_idx = sampled_or_fixed_sequence_idx.clone()
            x_to_gly_sampled_or_fixed_sequence_idx[sampled_x_residue_mask] = aa_short_to_idx['G']
            sampled_sequence_chi_masks = self.rotamer_builder.aa_to_chi_angle_mask[x_to_gly_sampled_or_fixed_sequence_idx] # type: ignore

            # Store sequence logits and sampled sequence indices in output tensors.
            output_tensors.sequence_logits[curr_decoded_indices] = curr_out_logits
            output_tensors.sampled_sequence_indices[curr_decoded_indices] = sampled_or_fixed_sequence_idx

            # Decode chi angles from final protein node embeddings.
            #   pull previously decoded chi angles from output tensors.
            curr_nodes_decoder_embeddings = prot_node_stack[-1].get_indices(curr_decoded_indices)
            prot_scalars = curr_nodes_decoder_embeddings.scalars
            chi_prev = torch.empty((curr_decoded_indices.shape[0], 0), device=self.device)
            for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
                # Predict chi angles from final protein node embeddings.
                chi_logits = chi_layer(torch.cat([prot_scalars, sequence_embeddings[curr_decoded_indices], chi_prev], dim=1))

                # Sample chi angles if temperature is specified, otherwise take argmax.
                if chi_angle_sample_temperature is None:
                    chi_sample = chi_logits.argmax(dim=-1)
                else:
                    chi_probs = torch.softmax(chi_logits / chi_angle_sample_temperature, dim=-1)
                    chi_sample = torch.distributions.Categorical(probs=chi_probs).sample()

                chi_sample_one_hot = F.one_hot(chi_sample, num_classes=self.rotamer_builder.num_chi_bins).float()
                chi_sample_offset = self.chi_offset_prediction_layers[chi_idx](torch.cat([chi_logits, chi_sample_one_hot], dim=1)).squeeze()

                # Convert sampled index to angle, then to RBF encoding.
                #   1 in chain mask tells us to sample chi angle, a 0 tells us to use the input chi angle.
                # sampled_angles = torch.remainder(self.rotamer_builder.index_to_degree_bin[chi_sample] + chi_sample_offset + 180, 360) - 180 # type: ignore
                # sampled_angles = (curr_chain_mask * input_chi_angles[curr_decoded_indices, chi_idx]) + ((1 - curr_chain_mask) * sampled_angles)

                # This is only 1 if we are both trying to fix the identity of the residue and the residue is not NaN
                nan_mask = ~input_chi_angles[curr_decoded_indices, chi_idx].isnan()
                sample_mask = (curr_chain_mask.bool() & nan_mask).long()
                # Keep sampled angles according to whether sample mask is 1 or 0.
                sampled_angles = (sample_mask * input_chi_angles[curr_decoded_indices, chi_idx].nan_to_num()) + ((1 - sample_mask) * sampled_angles)

                curr_chi_encoding = self.rotamer_builder.compute_binned_degree_basis_function(sampled_angles.unsqueeze(-1)).squeeze(1)
                chi_prev = torch.cat([chi_prev, curr_chi_encoding], dim=1)

                # Convert chi angle encoding to logits and store in output tensors.
                curr_chi_masks = sampled_sequence_chi_masks[:, chi_idx]
                output_tensors.chi_logits[curr_decoded_indices[curr_chi_masks], chi_idx] = chi_logits[curr_chi_masks]
                output_tensors.sampled_chi_encoding[curr_decoded_indices[curr_chi_masks], chi_idx] = curr_chi_encoding[curr_chi_masks]
                output_tensors.sampled_chi_degrees[curr_decoded_indices[curr_chi_masks], chi_idx] = sampled_angles[curr_chi_masks]

                if chi_idx == 3:
                    # Don't need to update node representations for the last chi angle since there is no next chi angle to predict.
                    break

                gvp_layer, gvp_norm = self.chi_vector_update_layers[chi_idx], self.chi_vector_layer_norms[chi_idx]
                concat_features = EquivariantData(torch.cat([prot_scalars, chi_prev], dim=1), curr_nodes_decoder_embeddings.vectors)
                curr_nodes_decoder_embeddings = gvp_norm(gvp_layer(concat_features))
                prot_scalars = curr_nodes_decoder_embeddings.scalars

        sampled_decoding_order = torch.cat(output_decoding_order, dim=0)
        transition_probabilities = torch.cat(transition_probabilities, dim=0)
        sampled_transition_entropies = torch.cat(sampled_transition_entropies, dim=0)
        if return_encoder_embeddings:
            return output_tensors, prot_nodes.scalars, pr_pr_eattr, lig_nodes.scalars, lig_pr_eattr, sampled_decoding_order, transition_probabilities, sampled_transition_entropies # type: ignore
        return output_tensors, sampled_decoding_order, transition_probabilities, sampled_transition_entropies


class LigandEncoderModule(nn.Module):
    """
    Implements a HomoGATv2 that operates over ligand nodes.
    """
    def __init__(self, cg_node_in_dim: int, node_embedding_dim: int , edge_embedding_dim: int, lig_lig_edge_rbf_params: dict, num_encoder_layers: int, num_vectors: int, **kwargs):
        super(LigandEncoderModule, self).__init__()
        self.num_vectors = num_vectors
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim

        self.ligand_input_layer = nn.Linear(cg_node_in_dim, node_embedding_dim)
        self.input_gvp = GVP((node_embedding_dim, 1), (node_embedding_dim, num_vectors), vector_gate=True)
        self.ligand_edge_rbf_encoding = RBF_Encoding(**lig_lig_edge_rbf_params)
        self.ligand_edge_input_layer = nn.Linear(self.ligand_edge_rbf_encoding.num_bins, edge_embedding_dim)

        # No need to update edges in the final layer since ligand edges aren't used downstream.
        self.gat_layers = nn.ModuleList([
            HomoGATv2(
                node_embedding_dim, 
                edge_embedding_dim, 
                num_vectors = self.num_vectors,
                update_edges = lidx < (num_encoder_layers - 1), 
                use_mlp_node_update = False,
                **kwargs
            ) for lidx in range(num_encoder_layers)
        ])

    def forward(self, ligand_data: LigandData) -> EquivariantData:

        # Appease the type checker.
        assert ligand_data.lig_lig_edge_distance is not None, "Ligand-ligand edge distance must be specified in ligand data."
        assert ligand_data.lig_lig_edge_index is not None, "Ligand-ligand edge index must be specified in ligand data."
        
        # Encode ligand nodes and edges with GAT layers.
        cg_nodes = self.ligand_input_layer(ligand_data.lig_nodes)

        # Initialize input vectors to be the vector pointing from current atom to mean atom of the neighbors of each node.
        avg_neighbor_coords = scatter(ligand_data.lig_coords[ligand_data.lig_lig_edge_index[0]], ligand_data.lig_lig_edge_index[1], dim=0, reduce='mean')
        input_vecs = (avg_neighbor_coords - ligand_data.lig_coords) 
        input_vecs = input_vecs / _norm_no_nan(input_vecs, axis=-1, keepdims=True)

        # Merge scalar and vector representations into a single tensor.
        cg_node_data = EquivariantData(cg_nodes, input_vecs.float())

        # Expand vectors to num_vectors dimension.
        # Doesn't seem to matter if cg_nodes are updated here or not.
        cg_node_data = self.input_gvp(cg_node_data)
        lig_lig_eattr = self.ligand_edge_input_layer(self.ligand_edge_rbf_encoding(ligand_data.lig_lig_edge_distance))
        
        # Apply graph attention layers to update nodes
        for gat_layer in self.gat_layers:
            cg_node_data, lig_lig_eattr = gat_layer(cg_node_data, lig_lig_eattr, ligand_data.lig_lig_edge_index)

        return cg_node_data
    

class LASErMPNN_Encoder(nn.Module):
    """
    Implements a HetGATv2 that operates over protein and ligand nodes.
    """
    def __init__(self, node_embedding_dim: int, edge_embedding_dim: int, num_vectors: int, num_attention_heads: int, atten_head_aggr_layers: int, dropout: float, **kwargs):
        super(LASErMPNN_Encoder, self).__init__()

        self.hetgat = HeteroGATv2(
            2, node_embedding_dim, edge_embedding_dim, atten_head_aggr_layers, 
            num_attention_heads, dropout, num_vectors=num_vectors,
            use_mlp_node_update=True, use_residual_node_update=True, compute_edge_updates=True, **kwargs
        )

        self.final_lig_pr_mlp = nn.Linear(edge_embedding_dim + num_vectors, edge_embedding_dim)
        self.final_pr_pr_mlp = nn.Linear(edge_embedding_dim + (num_vectors ** 2), edge_embedding_dim)

    def forward(self, prot_nodes: EquivariantData, lig_nodes: EquivariantData, pr_pr_eattr: torch.Tensor, lig_pr_eattr: torch.Tensor, batch: BatchData) -> torch.Tensor:
        # TODO: handle lig_nodes is None
        pr_node_update, (pr_pr_eattr, lig_pr_eattr) = self.hetgat( 
            prot_nodes, [(prot_nodes, prot_nodes), (lig_nodes, prot_nodes)], [pr_pr_eattr, lig_pr_eattr], [batch.pr_pr_edge_index, batch.lig_pr_edge_index]
        )

        ligand_displacement_dot_products = compute_frame_vector_ligand_displacement_vector_dot_product(batch.ligand_data.lig_coords, batch.backbone_coords, pr_node_update.vectors, batch.lig_pr_edge_index)
        lig_pr_eattr = self.final_lig_pr_mlp(torch.cat([lig_pr_eattr, ligand_displacement_dot_products], dim=1))

        frame_vector_dot_products = compute_protein_frame_vector_dot_product(pr_node_update.vectors, batch.pr_pr_edge_index)
        pr_pr_eattr = self.final_pr_pr_mlp(torch.cat([pr_pr_eattr, frame_vector_dot_products], dim=1))

        return pr_node_update, pr_pr_eattr, lig_pr_eattr # type: ignore
    

class LASErMPNN_Decoder(nn.Module):
    """
    Implements a HetGATv2 that operates over protein and ligand nodes with masking.
    """
    def __init__(
        self, 
        node_embedding_dim: int, edge_embedding_dim: int, chi_embedding_dim: int, 
        num_vectors: int, atten_head_aggr_layers: int, num_attention_heads: int, 
        dropout: float, **kwargs
    ):
        super(LASErMPNN_Decoder, self).__init__()

        decoder_masked_input_node_embedding = (2 * node_embedding_dim) + edge_embedding_dim + chi_embedding_dim

        self.hetgat = HeteroGATv2(
            num_subgraphs_to_merge = 2, 
            node_embedding_dim = [(decoder_masked_input_node_embedding, node_embedding_dim), (node_embedding_dim, node_embedding_dim)], 
            edge_embedding_dim = [0, edge_embedding_dim],
            atten_head_aggr_layers = atten_head_aggr_layers,
            num_attention_heads = num_attention_heads,
            dropout = dropout,
            use_mlp_node_update = True,
            use_residual_node_update = True,
            compute_edge_updates = False,
            output_node_embedding_dim = node_embedding_dim,
            output_edge_embedding_dim = edge_embedding_dim,
            num_vectors=num_vectors,
            **kwargs
        )
    
    def forward(
        self, source_node_edge_features_exp: torch.Tensor, prot_nodes: EquivariantData, 
        pr_pr_edge_index: torch.Tensor, lig_nodes: EquivariantData, lig_pr_eattr: torch.Tensor, 
        lig_pr_edge_index: torch.Tensor
    ) -> EquivariantData:
        """
        """
        # Pre-expand ligand -> protein nodes and edges as well.
        lig_source = lig_nodes.scalars[lig_pr_edge_index[0]]
        lig_sink = prot_nodes.scalars[lig_pr_edge_index[1]]
        prot_sink = prot_nodes.scalars[pr_pr_edge_index[1]]

        # Update protein nodes with masked edges.
        prot_nodes = self.hetgat.preexpanded_edges_forward(
            prot_nodes,
            [(source_node_edge_features_exp, prot_sink), (lig_source, lig_sink)],
            [pr_pr_edge_index, lig_pr_edge_index],
            [torch.empty((source_node_edge_features_exp.shape[0], 0), device=prot_nodes.device), lig_pr_eattr],
        )

        return prot_nodes


class RBF_Encoding(nn.Module):
    """
    Implements the RBF Encoding from ProteinMPNN as a module that can get stored in the model.
    """
    def __init__(self, num_bins: int, bin_min: float, bin_max: float):
        super(RBF_Encoding, self).__init__()
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.D_sigma =  (bin_max - bin_min) / num_bins
        self.register_buffer('D_mu', torch.linspace(bin_min, bin_max, num_bins).view([1,-1]))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances in last dimension to RBF encoding in an expanded (num_bins) dimension
            (N, M)  -->  (N, M, num_bins)
        """
        D_expand = torch.unsqueeze(distances, -1)
        rbf_encoding = torch.exp(-((D_expand - self.D_mu) / self.D_sigma)**2) + 1e-4
        return rbf_encoding


class SpiceDatasetPretrainingModule(nn.Module):
    """
    Applies the LASErMPNN ligand encoder to partial charge prediction.
    """
    def __init__(self, num_ligand_encoder_vectors: int, ligand_edge_embedding_dim: int, use_hydrogens: bool, **kwargs):
        super(SpiceDatasetPretrainingModule, self).__init__()
        self.num_vectors = num_ligand_encoder_vectors
        self.ligand_featurizer = LigandFeaturizer(use_hydrogens)

        self.ligand_encoder = LigandEncoderModule(
            self.ligand_featurizer.output_dim, 
            num_vectors=num_ligand_encoder_vectors, 
            edge_embedding_dim=ligand_edge_embedding_dim, 
            **kwargs
        )
        
        self.predict_dipoles_layer = DenseGVP((self.ligand_encoder.node_embedding_dim, num_ligand_encoder_vectors), (self.ligand_encoder.node_embedding_dim, num_ligand_encoder_vectors), (self.ligand_encoder.node_embedding_dim, 1), dropout=kwargs['dropout'], intermediate_norm=True)
        self.partial_charge_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, 1)
        self.mayer_order_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, 1)

        self.hybridization_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, len(POSSIBLE_HYBRIDIZATION_LIST))
        self.formal_charge_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, len(POSSIBLE_FORMAL_CHARGE_LIST))
        self.num_connected_hydrogens_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, len(POSSIBLE_NUM_HYDROGENS_LIST))
        self.possible_degree_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, len(POSSIBLE_DEGREE_LIST))
        self.is_aromatic_pred_layer = DenseMLP(self.ligand_encoder.node_embedding_dim, self.ligand_encoder.node_embedding_dim, len(POSSIBLE_IS_AROMATIC_LIST))

        self.gelu = nn.GELU()
    
    @property
    def device(self) -> torch.device:
        """
        Returns the device that the model is currently on when addressed as model.device
        """
        return next(self.parameters()).device

    def forward(self, batch: Union[SpiceBatchData, LigandData]):
        # Build up encoder representations.
        if isinstance(batch, SpiceBatchData):
            cg_nodes = self.ligand_encoder(batch.ligand_data)
        elif isinstance(batch, LigandData):
            cg_nodes = self.ligand_encoder(batch)
        else:
            raise ValueError("Input batch must be of type SpiceBatchData or LigandData.")

        pred_dipoles = self.predict_dipoles_layer(cg_nodes).vectors
        pred_partial_charges = self.partial_charge_pred_layer(self.gelu(cg_nodes.scalars))
        pred_mayer_order = self.partial_charge_pred_layer(self.gelu(cg_nodes.scalars))
        pred_hybridization = self.hybridization_pred_layer(self.gelu(cg_nodes.scalars))
        pred_formal_charge = self.formal_charge_pred_layer(self.gelu(cg_nodes.scalars))
        pred_num_connected_hydrogens = self.num_connected_hydrogens_pred_layer(self.gelu(cg_nodes.scalars))
        pred_possible_degree = self.possible_degree_pred_layer(self.gelu(cg_nodes.scalars))
        pred_is_aromatic = self.is_aromatic_pred_layer(self.gelu(cg_nodes.scalars))

        return pred_dipoles, pred_partial_charges, pred_mayer_order, pred_hybridization, pred_formal_charge, pred_num_connected_hydrogens, pred_possible_degree, pred_is_aromatic
