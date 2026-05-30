import numpy as np
import torch
from Bio.SeqUtils import seq3, seq1
from Bio.PDB import is_aa
from Bio.Align import PairwiseAligner
from model.utils.chemical import NTOTAL, RES_NB_JUMP
from utils import get_amino_acid_coords, one_letter_token


def process_apo(structure, protein_seq):
    all_coords = []
    all_bfac = []
    all_occ = []
    all_atom_mask = []
    all_seqs = []
    all_aa_token = []
    all_seq_mask = []
    all_res_nb = []
    all_chains = []
    all_icode = []
    res_num = 0


    for chain_i, chain in enumerate(structure[0]):
        residue_idx_list, coords_list, atom_mask, bfac_list, occ_list = get_amino_acid_coords(structure, chain_id=chain.id,
                                                                                              max_atom_num=NTOTAL,)

        res_seq = ''
        for residue in residue_idx_list:
            res_seq += seq1(residue.resname)

        apo_seq = str(protein_seq[chain_i].seq)

        alter_coords_list = np.zeros((len(apo_seq), NTOTAL, 3))
        alter_bfac_list = np.full((len(apo_seq), NTOTAL), np.nan)
        alter_occ_list = np.zeros((len(apo_seq), NTOTAL))
        alter_atom_mask = np.zeros((len(apo_seq), NTOTAL))
        seq_mask = np.ones(len(apo_seq))
        aa_tokens = []

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -2
        # 让开 gap 很贵，避免为了解决首位不匹配而开一个 gap
        aligner.open_gap_score = -12
        aligner.extend_gap_score = -1
        aln = aligner.align(res_seq, apo_seq)[0]
        aln_res_seq = aln[0]

        residue_idx = 0
        # 根据含有gap的对齐解析序列进行修正
        for i, seq_residue in enumerate(aln_res_seq):
            if seq_residue != '-':
                if residue_idx < len(residue_idx_list):
                    alter_coords_list[i] = coords_list[residue_idx]
                    alter_bfac_list[i] = bfac_list[residue_idx]
                    alter_occ_list[i] = occ_list[residue_idx]
                    alter_atom_mask[i] = atom_mask[residue_idx]
                    all_icode.append(residue_idx_list[residue_idx].full_id[-1][2])
                    residue_idx += 1
            else:
                all_icode.append(' ')
            if not is_aa(seq3(apo_seq[i])):
                seq_mask[i] = 0.
                aa_tokens.append(20)
            else:
                aa_tokens.append(one_letter_token[apo_seq[i]])

        all_coords.append(alter_coords_list)
        all_bfac.append(alter_bfac_list)
        all_occ.append(alter_occ_list)
        all_atom_mask.append(alter_atom_mask)
        all_aa_token.extend(aa_tokens)
        all_seqs.append(apo_seq)
        all_seq_mask.append(seq_mask)
        all_res_nb.append(np.arange(1 + res_num, len(apo_seq) + 1 + res_num))
        all_chains.extend([chain_i] * len(apo_seq))

        res_num += (len(apo_seq) + RES_NB_JUMP)

    return {
        'apo_xyz': torch.from_numpy(np.concatenate(all_coords)).float(),
        'apo_bfac': torch.from_numpy(np.concatenate(all_bfac)).float(),
        'apo_occ': torch.from_numpy(np.concatenate(all_occ)).float(),
        'apo_xyz_mask': torch.from_numpy(np.concatenate(all_atom_mask)).bool(),
        'apo_seq': all_seqs,
        'apo_aa_token': torch.tensor(all_aa_token),
        'apo_aa_mask': torch.from_numpy(np.concatenate(all_seq_mask)).bool(),
        'apo_icode': all_icode,
        'apo_chain': all_chains,
        'apo_res_nb': torch.from_numpy(np.concatenate(all_res_nb)),
    }
