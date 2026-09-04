"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0

----------------
This file may have been modified by Zhao Kailong et al.
"""
import torch
from torch import nn
from src.models.full_atom.utils.all_atom import atom14_to_atom37
from openfold.utils import rigid_utils as ru
from tqdm import tqdm
from src.models.full_atom.score_network import BaseScoreNetwork
from torch import einsum

import numpy as np
import os
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial.transform import Rotation

from openfold.data import data_transforms
from openfold.np import residue_constants as rc
from openfold.np.protein import Protein
from openfold.np import protein
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from src.utils import hydra_utils

logger = hydra_utils.get_pylogger(__name__)
pdb_parser = PDBParser(QUIET=True)

class GuidanceScoreNetwork(BaseScoreNetwork):
    def __init__(
            self,
            cond_model_nn,
            cond_ckpt_path,
            cfg,
            msta_dir,
            uncond_model_nn=None,
            uncond_ckpt_path=None,
            load_strict: bool = True,
            **kwargs
    ):
        super(GuidanceScoreNetwork, self).__init__(cond_model_nn, cfg)
        self.diffuser = None
        self.load_strict = load_strict

        if cond_ckpt_path:
            logger.info(f"✅ Successfully loaded conditional model weights in strict mode (including the force-field head): {cond_ckpt_path}")
            cond_ckpt = torch.load(cond_ckpt_path, map_location="cpu")["state_dict"]
            cond_state_dict = {}
            prefix = "score_network.model_nn."
            for key, value in cond_ckpt.items():
                if key.startswith(prefix):
                    cond_state_dict[key[len(prefix):]] = value

            logger.info(f"load_state_dict(strict={self.load_strict}) for cond_model_nn")
            self.model_nn.load_state_dict(cond_state_dict, strict=self.load_strict)
            del cond_ckpt

        for param in self.model_nn.parameters():
            param.requires_grad = False

        self.uncond_model_nn = uncond_model_nn
        self.msta_dir = msta_dir

        if uncond_ckpt_path:
            logger.info(f"✅ Successfully loaded unconditional model weights in strict mode (including the force-field head): {uncond_ckpt_path}")
            uncond_ckpt = torch.load(uncond_ckpt_path, map_location="cpu")["state_dict"]
            uncond_state_dict = {}
            prefix = "score_network.model_nn."
            for key, value in uncond_ckpt.items():
                if key.startswith(prefix):
                    uncond_state_dict[key[len(prefix):]] = value

            logger.info(f"load_state_dict(strict={self.load_strict}) for uncond_model_nn")
            self.uncond_model_nn.load_state_dict(uncond_state_dict, strict=self.load_strict)
            del uncond_ckpt
    ### When training the force fine-tuning model, set strict=False to load the base model;
    ### When using the trained fine-tuned model for inference, set strict=True;

    @property
    def device(self):
        return self.model_nn.device

    def forward(
        self,
        aatype,
        t,
        rigids_t,
        rigids_mask,
        padding_mask,
        gt_feat,
        res_idx=None,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ):
        self.diffuser._so3_diffuser.use_cached_score = False
        rigids_t = ru.Rigid.from_tensor_7(rigids_t)
        rigids_mask = rigids_mask * padding_mask

        model_out = self.model_nn(
            aatype=aatype,
            padding_mask=padding_mask,
            t=t,
            rigids_t=rigids_t,
            rigids_mask=rigids_mask,
            res_idx=res_idx,
            pretrained_node_repr=pretrained_node_repr,
            pretrained_edge_repr=pretrained_edge_repr,
        )

        input_feat = {
            "aatype": aatype,
            "t": t,
            "rigids_t": rigids_t,
            "rigids_mask": rigids_mask,
            "padding_mask": padding_mask,
            "gt_feat": gt_feat,
            "res_idx": res_idx,
            "pretrained_node_repr": pretrained_node_repr,
            "pretrained_edge_repr": pretrained_edge_repr,
        }

        loss = self.loss_fn(input_feat, model_out)
        return loss, {"total": loss.item()}

    def loss_fn(self, input_feat, model_out, **kwargs):
        return torch.tensor(0)

    def forward_guidance_model(self, input_feat):
        model_out = self.model_nn(**input_feat)
        return model_out


    def reverse_sample_stage1(
            self,
            aatype,
            padding_mask,
            chain_name,
            output_dir,
            pretrained_node_repr=None,
            pretrained_edge_repr=None,
            **kwargs,
    ):
        assert not self.model_nn.training
        self.diffuser._so3_diffuser.use_cached_score = True

        """ Reverse sampling. """
        rigids_mask = padding_mask.float()

        batch_size, seq_len = aatype.shape[:2]

        msta_fpath = os.path.join(self.msta_dir, f"{chain_name[0]}.msta")

        # [MODIFIED] folding_priority shape is now [B, L] or [Num_Paths, L]
        # We ensure it matches batch_size
        folding_priority = self.load_msta(msta_fpath, seq_len).to(aatype.device)

        # If batch_size matches folding_priority columns, use it directly.
        # Otherwise, slice or repeat as needed (assuming strictly matched based on your previous script)
        if folding_priority.shape[0] != batch_size:
             # Basic handling: if priority has more, slice; if less, repeat (or error)
             if folding_priority.shape[0] > batch_size:
                 folding_priority = folding_priority[:batch_size]
             else:
                 raise ValueError(f"Batch size ({batch_size}) > MSTA columns ({folding_priority.shape[0]}).")

        base_dt = 1.0 / (self.cfg.stage1_diffusion_steps + self.cfg.base_steps)
        dt_scale_coeff = ((self.cfg.stage1_diffusion_steps + self.cfg.base_steps) / self.cfg.base_steps) - 1

        # [MODIFIED] No unsqueeze needed, folding_priority is [B, L]
        dt_scale_first = 1 + dt_scale_coeff * folding_priority
        dt = base_dt * dt_scale_first  # [B, L]

        t = torch.ones(batch_size, seq_len, device=aatype.device)  # [B, L]

        rigids_t = self.diffuser.sample_ref(
            n_samples=batch_size, seq_len=seq_len, device=aatype.device
        )

        for step_t in tqdm(range(self.cfg.stage1_diffusion_steps + self.cfg.base_steps)):
            input_feat = {
                "aatype": aatype,
                "t": t,
                "rigids_t": rigids_t,
                "rigids_mask": rigids_mask,
                "padding_mask": padding_mask,
                "pretrained_node_repr": pretrained_node_repr,
                "pretrained_edge_repr": pretrained_edge_repr,
            }

            model_out = self.forward_guidance_model(
                input_feat=input_feat,
            )

            active_mask = (t > 0.01).float()
            current_rigids_mask = rigids_mask * active_mask

            pred_rot_score = (
                    self.diffuser.calc_rot_score(
                        rigids_t.get_rots(),
                        model_out["pred_rigids_0"].get_rots(),
                        t,
                    )
                    * current_rigids_mask[..., None]
            )

            pred_trans_score = (
                    self.diffuser.calc_trans_score(
                        rigids_t.get_trans(),
                        model_out["pred_rigids_0"].get_trans(),
                        t,
                        use_torch=True,
                    )
                    * current_rigids_mask[..., None]
            )

            rigids_s = self.diffuser.reverse(
                rigids_t=rigids_t,
                rot_score=pred_rot_score,
                trans_score=pred_trans_score,
                t=t,
                dt=dt,
                diffuse_mask=active_mask
            )

            rigids_t = rigids_s

            update_mask = (t > 0.01).float()
            t = t - dt * update_mask
            t = torch.clamp(t, min=0.01)

            output_dir_allatom = os.path.join(output_dir, "stage1")
            os.makedirs(output_dir_allatom, exist_ok=True)

            all_frames_to_global = self.torsion_angles_to_frames(
                rigids_t,
                model_out["pred_torsions"],
                aatype,
            )
            model_out["pred_atom14"] = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                torch.fmod(aatype, 20),
            )
            pred_atom37, atom37_mask = atom14_to_atom37(model_out["pred_atom14"], aatype)
            if step_t - self.cfg.base_steps >= 0:
                for i in range(batch_size):
                    padding_mask_i = padding_mask[i]
                    aatype_i = aatype[i][padding_mask_i].cpu().numpy()
                    atom37_i = pred_atom37[i][padding_mask_i].detach().cpu().numpy()
                    atom37_mask_i = atom37_mask[i][padding_mask_i].detach().cpu().numpy()
                    res_idx = np.arange(aatype_i.shape[0])
                    gen_protein = Protein(
                        aatype=aatype_i,
                        atom_positions=atom37_i,
                        atom_mask=atom37_mask_i,
                        residue_index=res_idx + 1,
                        chain_index=np.zeros_like(aatype_i),
                        b_factors=np.zeros_like(atom37_mask_i),
                    )
                    # sample index corresponds to MSTA column
                    pdb_path = os.path.join(output_dir_allatom, f"stage1_step_{step_t - self.cfg.base_steps}_sample{i}.pdb")
                    with open(pdb_path, "w") as fp:
                        fp.write(protein.to_pdb(gen_protein))


    def reverse_sample_stage2(
            self,
            aatype,
            padding_mask,
            chain_name,
            output_dir,
            pretrained_node_repr=None,
            pretrained_edge_repr=None, **kwargs,
    ):
        assert not self.model_nn.training
        self.diffuser._so3_diffuser.use_cached_score = True

        """ Reverse sampling (modified version with three-region guidance). """
        rigids_mask = padding_mask.float()

        batch_size, seq_len = aatype.shape[:2]
        base_dt = 1.0 / self.cfg.stage2_diffusion_steps

        for step_num in tqdm(
                range(self.cfg.starting_steps + self.cfg.base_steps, self.cfg.final_steps + self.cfg.base_steps)):
            logger.info(f"step {step_num}")

            stage1_pdb_dir = os.path.join(output_dir, "stage1")
            current_step_in_stage1 = step_num - self.cfg.base_steps

            dynamic_weight_list = []
            current_pdb_paths = []

            for i in range(batch_size):
                current_pdb_path = os.path.join(stage1_pdb_dir, f"stage1_step_{current_step_in_stage1}_sample{i}.pdb")
                final_stage1_pdb_path = os.path.join(stage1_pdb_dir, f"stage1_step_{self.cfg.stage1_diffusion_steps - 1}_sample{i}.pdb")

                current_pdb_paths.append(current_pdb_path)

                for pdb_path in [current_pdb_path, final_stage1_pdb_path]:
                    if not os.path.exists(pdb_path):
                        raise FileNotFoundError(f"The stage1 pdb file does not exist: {pdb_path}")

                ca_coords_current = self._get_ca_coords(current_pdb_path, seq_len, aatype)
                ca_coords_final = self._get_ca_coords(final_stage1_pdb_path, seq_len, aatype)

                distances = torch.norm(ca_coords_current - ca_coords_final, dim=1)

                dw = torch.full(
                    size=(seq_len,),
                    fill_value=1 - self.cfg.clsfree_guidance_strength,
                    device=aatype.device
                )

                cond1_mask = distances < 1.0
                cond1_segments = self._find_continuous_segments(cond1_mask, seq_len)
                for start, end in cond1_segments:
                    dw[start:end+1] = self.cfg.clsfree_guidance_strength

                cond2_mask = (distances > 1.0) & (distances <= 5.0)
                cond2_segments = self._find_continuous_segments(cond2_mask, seq_len)
                for start, end in cond2_segments:
                    for k in range(start, end+1):
                        if dw[k] == (1 - self.cfg.clsfree_guidance_strength):
                            dw[k] = 0.5

                dynamic_weight_list.append(dw)

            dynamic_weight = torch.stack(dynamic_weight_list)  # [B, L]
            dynamic_weight = torch.clamp(dynamic_weight, min=0.0, max=1.0)

            t = torch.ones(batch_size, seq_len, device=aatype.device)  # [B, L]
            dt = base_dt * torch.ones_like(t)  # [B, L]

            logger.info(f"Initialize rigids_t from the stage1 pdbs (Batch size: {batch_size})")

            rigids_list = []
            for i, pdb_path in enumerate(current_pdb_paths):
                rigid_i = self.initialize_rigids_from_pdb(
                    pdb_path=pdb_path,
                    seq_len=seq_len,
                    device=aatype.device,
                    batch_size=1,
                    aatype=aatype[i:i+1]
                )
                rigids_list.append(rigid_i)

            rots_all = torch.cat([r.get_rots().get_rot_mats() for r in rigids_list], dim=0)
            trans_all = torch.cat([r.get_trans() for r in rigids_list], dim=0)

            rot_obj = ru.Rotation(rot_mats=rots_all)
            rigids_t = ru.Rigid(rots=rot_obj, trans=trans_all)

            for step_t in tqdm(range(self.cfg.stage2_diffusion_steps), leave=False):
                active_mask = (t > 0.01).float()
                current_rigids_mask = rigids_mask * active_mask

                input_feat = {
                    "aatype": aatype,
                    "t": t,
                    "rigids_t": rigids_t,
                    "rigids_mask": rigids_mask,
                    "padding_mask": padding_mask,
                    "pretrained_node_repr": pretrained_node_repr,
                    "pretrained_edge_repr": pretrained_edge_repr,
                }

                cond_pred_force_t, model_out = self.forward_guidance_model(input_feat=input_feat)

                cond_pred_rot_score = (
                        self.diffuser.calc_rot_score(rigids_t.get_rots(), model_out["pred_rigids_0"].get_rots(), t)
                        * current_rigids_mask[..., None]
                )
                cond_pred_trans_score = (
                        self.diffuser.calc_trans_score(rigids_t.get_trans(), model_out["pred_rigids_0"].get_trans(), t,
                                                       use_torch=True)
                        * current_rigids_mask[..., None]
                )

                cond_guided_trans = cond_pred_trans_score - self.cfg.force_guidance_strength * cond_pred_force_t

                uncond_pred_rot_score = torch.zeros_like(cond_pred_rot_score)
                uncond_guided_trans = torch.zeros_like(cond_pred_trans_score)

                if self.uncond_model_nn and self.cfg.clsfree_guidance_strength <= 1:
                    uncond_model_out = self.uncond_model_nn(
                        aatype=aatype, padding_mask=padding_mask, t=t, rigids_t=rigids_t,
                        rigids_mask=rigids_mask, res_idx=None,
                        pretrained_node_repr=None, pretrained_edge_repr=None,
                    )

                    u_force_t = torch.zeros_like(cond_pred_force_t)
                    if "pred_force_t" in uncond_model_out and "pred_force_0" in uncond_model_out:
                        u_f_t = torch.nn.functional.normalize(uncond_model_out["pred_force_t"], dim=-1, p=2)
                        u_f_0 = torch.nn.functional.normalize(uncond_model_out["pred_force_0"], dim=-1, p=2)
                        t_ext = t[..., None]
                        u_force_t = (1.0 - t_ext) * u_f_0 + (t_ext * (1.0 - t_ext)) * u_f_t

                    uncond_pred_rot_score = (
                            self.diffuser.calc_rot_score(rigids_t.get_rots(),
                                                         uncond_model_out["pred_rigids_0"].get_rots(), t)
                            * current_rigids_mask[..., None]
                    )
                    uncond_pred_trans_score = (
                            self.diffuser.calc_trans_score(rigids_t.get_trans(),
                                                           uncond_model_out["pred_rigids_0"].get_trans(), t,
                                                           use_torch=True)
                            * current_rigids_mask[..., None]
                    )

                    uncond_guided_trans = uncond_pred_trans_score - self.cfg.force_guidance_strength * u_force_t

                target_shape = cond_pred_rot_score.shape
                dynamic_weight_expanded = dynamic_weight.view(*dynamic_weight.shape, 1).expand(target_shape)
                w = dynamic_weight_expanded * self.cfg.clsfree_guidance_strength

                final_rot_score = w * cond_pred_rot_score + (1 - w) * uncond_pred_rot_score

                final_trans_score = w * cond_guided_trans + (1 - w) * uncond_guided_trans

                rigids_s = self.diffuser.reverse(
                    rigids_t=rigids_t,
                    rot_score=final_rot_score,
                    trans_score=final_trans_score,
                    t=t,
                    dt=dt,
                    diffuse_mask=active_mask
                )
                rigids_t = rigids_s

                update_mask = (t > 0.01).float()
                t = t - dt * update_mask
                t = torch.clamp(t, min=0.01)


            output_dir_allatom = os.path.join(output_dir, "stage2")
            os.makedirs(output_dir_allatom, exist_ok=True)

            if "pred_atom14" not in model_out:
                all_frames_to_global = self.torsion_angles_to_frames(
                    rigids_t,
                    model_out["pred_torsions"],
                    aatype,
                )
                model_out["pred_atom14"] = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    torch.fmod(aatype, 20),
                )

            pred_atom37, atom37_mask = atom14_to_atom37(model_out["pred_atom14"], aatype)

            sample_info = []

            for i in range(batch_size):
                padding_mask_i = padding_mask[i]
                aatype_i = aatype[i][padding_mask_i].cpu().numpy()
                atom37_i = pred_atom37[i][padding_mask_i].detach().cpu().numpy()
                atom37_mask_i = atom37_mask[i][padding_mask_i].detach().cpu().numpy()
                res_idx = np.arange(aatype_i.shape[0])
                gen_protein = Protein(
                    aatype=aatype_i,
                    atom_positions=atom37_i,
                    atom_mask=atom37_mask_i,
                    residue_index=res_idx + 1,
                    chain_index=np.zeros_like(aatype_i),
                    b_factors=np.zeros_like(atom37_mask_i),
                )
                pdb_path = os.path.join(output_dir_allatom,
                                        f"Step_{step_num - self.cfg.base_steps}_sample{i}.pdb")
                with open(pdb_path, "w") as fp:
                    fp.write(protein.to_pdb(gen_protein))

                clash_count = self.calculate_backbone_clashes(pdb_path)
                sample_info.append((pdb_path, clash_count))


    def _get_ca_coords(self, pdb_path, seq_len, aatype):
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("prot", pdb_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDB {pdb_path}: {str(e)}")

        ca_coords = []
        residue_ids = []
        for chain in structure.get_chains():
            for res in chain.get_residues():
                if res.get_id()[0] == ' ':
                    try:
                        ca_atom = res['CA']
                        ca_coords.append(ca_atom.get_coord())
                        residue_ids.append(res.get_id()[1])
                    except KeyError:
                        raise ValueError(f"Residue {res.get_id()} missing CA atom（PDB: {pdb_path}）")

        if len(ca_coords) != seq_len:
            raise ValueError(f"PDB {pdb_path} contains {len(ca_coords)} valid residues, expected to contain {seq_len}")

        return torch.tensor(ca_coords, dtype=torch.float32, device=aatype.device)

    def _find_continuous_segments(self, mask, seq_len, min_length=3):
        segments = []
        start_idx = None
        for i in range(seq_len):
            if mask[i] and start_idx is None:
                start_idx = i
            elif not mask[i] and start_idx is not None:
                if (i - 1) - start_idx + 1 >= min_length:
                    segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None and (seq_len - 1) - start_idx + 1 >= min_length:
            segments.append((start_idx, seq_len - 1))
        return segments

    def calculate_backbone_clashes(self, pdb_path):
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", pdb_path)
        except:
            return float('inf')

        backbone_atoms = []
        for atom in structure.get_atoms():
            if atom.name in ['N', 'CA', 'C', 'O'] and atom.element != 'H':
                backbone_atoms.append(atom)

        if len(backbone_atoms) == 0:
            return 0

        ns = NeighborSearch(backbone_atoms)
        clash_count = 0
        vdw_radii = {'N': 1.55, 'CA': 1.7, 'C': 1.7, 'O': 1.52}

        for i, atom1 in enumerate(backbone_atoms):
            neighbors = ns.search(atom1.coord, 5.0, level='A')
            for atom2 in neighbors:
                if id(atom1) >= id(atom2):
                    continue
                res1 = atom1.get_parent()
                res2 = atom2.get_parent()
                chain1 = res1.get_parent()
                chain2 = res2.get_parent()

                if (res1.get_id()[1] == res2.get_id()[1] and chain1.id == chain2.id):
                    continue
                if abs(res1.get_id()[1] - res2.get_id()[1]) <= 1 and chain1.id == chain2.id:
                    continue

                r1 = vdw_radii.get(atom1.name, 1.7)
                r2 = vdw_radii.get(atom2.name, 1.7)
                distance = np.linalg.norm(atom1.coord - atom2.coord)

                if distance < 0.75 * (r1 + r2):
                    clash_count += 1
        return clash_count

    def initialize_rigids_from_pdb(self, pdb_path, seq_len, device, batch_size, aatype):
        atom_coords = self.load_pdb(pdb_path, seq_len)
        atom_coords -= np.nanmean(atom_coords, axis=(0, 1), keepdims=True)

        all_atom_positions_single = torch.from_numpy(atom_coords).to(device)
        all_atom_mask_single = torch.all(~torch.isnan(all_atom_positions_single), dim=-1)
        all_atom_positions_single = torch.nan_to_num(all_atom_positions_single, nan=0.0)

        aatype_seq_single = torch.LongTensor([
            rc.restype_order_with_x.get(res.item(), 20) for res in aatype[0]
        ]).to(device)

        openfold_feat_dict = {
            "aatype": aatype_seq_single.long(),
            "all_atom_positions": all_atom_positions_single.double(),
            "all_atom_mask": all_atom_mask_single.double(),
        }

        openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
        openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
        openfold_feat_dict = data_transforms.atom37_to_torsion_angles()(openfold_feat_dict)

        rigids_single = ru.Rigid.from_tensor_4x4(openfold_feat_dict["rigidgroups_gt_frames"])[:, 0]
        rots_tensor_single = rigids_single.get_rots().get_rot_mats()
        trans_single = rigids_single.get_trans()

        rots_tensor_batch = rots_tensor_single.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        trans_batch = trans_single.unsqueeze(0).expand(batch_size, -1, -1).clone()

        rots_batch = ru.Rotation(rot_mats=rots_tensor_batch)
        rigids_batch = ru.Rigid(rots=rots_batch, trans=trans_batch)

        return rigids_batch

    def load_msta(self, fold_priority_file: str, seq_len: int) -> torch.Tensor:
        # [MODIFIED] Read multi-column MSTA file
        priorities = []
        with open(fold_priority_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Assuming format: Index Score1 Score2 ... ScoreN
                if len(parts) >= 2:
                    # Read all scores for this residue position
                    row_scores = [float(x) for x in parts[1:]]
                    priorities.append(row_scores)

        # Convert to tensor. Shape will be [Seq_Len, Num_Paths]
        tensor = torch.tensor(priorities, dtype=torch.float32)

        # Truncate to seq_len if file is longer
        tensor = tensor[:seq_len]

        # Transpose to [Num_Paths, Seq_Len] so it matches [Batch, Len] convention
        return tensor.t()

    def load_pdb(self, pdb_path, seqlen=None):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("", pdb_path)
        atom_coords = np.full((seqlen, rc.atom_type_num, 3), np.nan, dtype=np.float32)
        model = struct[0]
        chain = next(model.get_chains())

        for residue in chain:
            res_id = residue.id[1]
            seq_idx = res_id - 1
            if seq_idx >= seqlen:
                continue
            for atom in residue:
                if atom.name in rc.atom_order:
                    atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord
        return atom_coords

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer("default_frames",
                                 torch.tensor(rc.restype_rigid_group_default_frame, dtype=float_dtype, device=device,
                                              requires_grad=False), persistent=False)
        if not hasattr(self, "group_idx"):
            self.register_buffer("group_idx",
                                 torch.tensor(rc.restype_atom14_to_rigid_group, device=device, requires_grad=False),
                                 persistent=False)
        if not hasattr(self, "atom_mask"):
            self.register_buffer("atom_mask", torch.tensor(rc.restype_atom14_mask, dtype=float_dtype, device=device,
                                                           requires_grad=False), persistent=False)
        if not hasattr(self, "lit_positions"):
            self.register_buffer("lit_positions",
                                 torch.tensor(rc.restype_atom14_rigid_group_positions, dtype=float_dtype, device=device,
                                              requires_grad=False), persistent=False)

    def torsion_angles_to_frames(self, r, alpha, f):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(r, f, self.default_frames, self.group_idx, self.atom_mask,
                                                             self.lit_positions)



class ForceGuidance(GuidanceScoreNetwork):
    def __init__(
        self,
        cond_model_nn,
        cond_ckpt_path,
        cfg,
        msta_dir=None,
        uncond_model_nn=None,
        uncond_ckpt_path=None,
        load_strict: bool = False,
        **kwargs,
    ):
        super(ForceGuidance, self).__init__(
            cond_model_nn,
            cond_ckpt_path,
            cfg,
            msta_dir=msta_dir,
            uncond_model_nn=uncond_model_nn,
            uncond_ckpt_path=uncond_ckpt_path,
            load_strict=load_strict,
            **kwargs,
        )
        for param in self.model_nn.structure_module.pred_force_t_net.parameters():
            param.requires_grad = True
        if self.model_nn.structure_module.pred_force_0:
            for param in self.model_nn.structure_module.pred_force_0_net.parameters():
                param.requires_grad = True

    def loss_fn(
        self,
        input_feat,
        model_out,
        **kwargs,
    ):

        pred_force_0 = model_out["pred_force_0"]
        pred_force_t = model_out["pred_force_t"]

        gt_feat = input_feat["gt_feat"]
        trans_0 = gt_feat["rigids_0"][..., 4:]
        trans_t = input_feat["rigids_t"].get_trans()
        t = input_feat["t"]
        rigids_mask = input_feat["rigids_mask"]

        pred_trans_score = (
            self.diffuser.calc_trans_score(
                trans_t,
                model_out["pred_rigids_0"].get_trans(),
                t,
                use_torch=True,
            )
            * rigids_mask[..., None]
        )

        energy_weight = torch.exp(-gt_feat["gt_energy_0"])
        gt_trans_score = gt_feat["trans_score"]

        trans_score_norm_scalar = gt_feat["trans_score_norm"][:, 0, 0]  # Shape:[B]
        sigma_t = 1.0 / trans_score_norm_scalar  # Shape: [B]

        seqlen = gt_trans_score.shape[1]

        mu_t = einsum(
            "a,blc->ablc", torch.sqrt(1.0 - sigma_t**2), trans_0
        )  # (B, B, L, 3)
        q_xt_x0 = torch.exp(
            -0.5 * (gt_trans_score**2).sum(dim=(-1, -2)) * sigma_t**2 / seqlen
        )  # (B,)

        zeta = (
            pred_trans_score[:, None, ...]
            + (trans_t[:, None, ...] - mu_t) * (1.0 / sigma_t**2)[:, None, None, None]
        )  # (B, B, L, 3)

        numerator = einsum(
            "a, b, abcd->abcd", q_xt_x0, energy_weight, zeta
        )  # (B, B, L, 3)
        denominator = q_xt_x0 * energy_weight.sum()  # (B,)
        gt_force_t = numerator.sum(dim=1) / denominator[:, None, None]  # (B, L, 3)

        # clamp norm
        gt_force_t_norm = torch.linalg.norm(gt_force_t, dim=-1)
        gt_force_t = (
            gt_force_t
            * (torch.clip(gt_force_t_norm, 0, 100) / (gt_force_t_norm + 1e-8))[
                ..., None
            ]
        )

        # force matching
        t_expand = t[..., None]
        pred_force_t = (1.0 - t_expand) * pred_force_0 + ((1.0 - t_expand) * t_expand) * pred_force_t


        scale = 0.1

        loss_t = torch.nn.functional.smooth_l1_loss(
            pred_force_t * scale,
            (0.1 * gt_force_t) * scale,
            reduction='none'
        )
        loss_0 = torch.nn.functional.smooth_l1_loss(
            pred_force_0 * scale,
            gt_feat["gt_force_0"] * scale,
            reduction='none'
        )
        force_score_mse = (loss_t + loss_0) * rigids_mask[..., None]

        force_loss = torch.sum(force_score_mse, dim=(-2, -1)) / (
            rigids_mask.sum(dim=-1) + 1e-6
        )

        return force_loss.mean()

    def forward_guidance_model(
            self,
            input_feat,
    ):
        model_out = self.model_nn(**input_feat)
        t = input_feat["t"]

        if "pred_force_t" in model_out and "pred_force_0" in model_out:
            pred_force_t = torch.nn.functional.normalize(
                model_out["pred_force_t"], dim=-1, p=2
            )
            pred_force_0 = torch.nn.functional.normalize(
                model_out["pred_force_0"], dim=-1, p=2
            )
            t_expand = t[..., None]  # [B, L, 1]
            pred_force_res = (1.0 - t_expand) * pred_force_0 + ((1.0 - t_expand) * t_expand) * pred_force_t
        else:
            pred_force_res = torch.zeros_like(model_out["pred_rigids_0"].get_trans())

        return pred_force_res, model_out
