import copy
import numpy as np
import torch
from Bio.SeqUtils import seq3
from Bio.PDB import is_aa
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from model.utils.chemical import NTOTAL, RES_NB_JUMP
from utils import load_config, get_pdb_structure, get_protein_res, notconsidered, get_amino_acid_coords, one_letter_token, \
    find_nearest_atom, get_pdb_links


def process_holo(select_structure, protein_seq, pocket_res_list, ligand_names, ligand_pt):
    config, config_name = load_config('../../config/base_config.yaml')

    all_coords = []
    all_bfac = []
    all_occ = []
    all_atom_mask = []
    all_seqs = []
    all_aa_token = []
    all_seq_mask = []
    all_res_nb = []
    all_chains = []
    all_chain_idx = []
    all_icode = []
    cova_atoms = []
    res_num = 0
    ligand_covalent_bonds = []
    is_leaving_ligand = [1 for _ in range(len(ligand_pt['ligand_atom_coords']))]  # 用于标记配体中的离去基团
    rcsb_holo_path = config['dataset']['base_path'] + '/holo_rcsb/%s.cif' % select_structure.id

    link_dict_list = get_pdb_links(rcsb_holo_path)
    if len(link_dict_list) != 0:
        structure = get_pdb_structure(config['dataset']['base_path'] + '/holo_rcsb/%s.cif' % select_structure.id)  # 读取含有共价键信息的结构文件
        for item in link_dict_list:
            """只能从pdb/cif文件读取"""
            if item['res_name1'] in ligand_names and is_aa(item['res_name2']):  # 必须保证另外一个是氨基酸
                ligand_chain = structure[0][item['chain_id1']]
                try:
                    ligand = ligand_chain[('H_%s' % item['res_name1'], int(item['res_seq1']), ' ')]
                except:  # 配体实际上是短肽的情况
                    ligand = ligand_chain[(' ', int(item['res_seq1']), ' ')]
                coval_ligand_atom = ligand[item['at_name1']]  # 在结构文件中记录的与蛋白质残基形成共价键的原子
                try:
                    coval_res = get_protein_res(rcsb_holo_path, int(item['res_seq2']), chain_id=item['chain_id2'])
                except:
                    continue
                coval_res_atom = coval_res.child_dict[item['at_name2']]  # 在结构文件中记录的形成共价键的蛋白质侧的原子
                if '%s-%s' % (coval_res_atom.element, coval_ligand_atom.element) in notconsidered:  # 排除特定的共价键
                    continue
                coval_atom_idx = -1  # 实际上形成共价键的配体原子索引（从0开始）
                for atom_idx, atom_xyz in enumerate(ligand_pt['ligand_atom_coords']):
                    if round(atom_xyz[0].item(), 4) == round(float(coval_ligand_atom.coord[0]), 4):  # 保留前四位小数
                        coval_atom_idx = atom_idx
                        break
                for atom_idx, atom_xyz in enumerate(ligand_pt['ligand_atom_coords']):
                    for pdb_atom_xyz in ligand:
                        if round(atom_xyz[0].item(), 4) == round(float(pdb_atom_xyz.coord[0]), 4):
                            is_leaving_ligand[atom_idx] = 0
                ligand_covalent_bonds.append([
                    [item['chain_id2'], coval_res_atom, item['at_name2']],
                    ['ligand', coval_atom_idx], coval_res
                ])
            elif item['res_name2'] in ligand_names and is_aa(item['res_name1']):  # 必须保证另外一个是氨基酸
                ligand_chain = structure[0][item['chain_id2']]
                try:
                    ligand = ligand_chain[('H_%s' % item['res_name2'], int(item['res_seq2']), ' ')]  # 在结构文件中记录的配体坐标信息
                except:  # 配体实际上是短肽的情况
                    ligand = ligand_chain[(' ', int(item['res_seq2']), ' ')]
                coval_ligand_atom = ligand[item['at_name2']]  # 在结构文件中记录的形成共价键的配体侧的原子
                try:
                    coval_res = get_protein_res(rcsb_holo_path, int(item['res_seq1']), chain_id=item['chain_id1'])
                except:
                    continue
                coval_res_atom = coval_res.child_dict[item['at_name1']]  # 在结构文件中记录的形成共价键的蛋白质侧的原子
                if '%s-%s' % (coval_res_atom.element, coval_ligand_atom.element) in notconsidered:  # 排除特定的共价键
                    continue
                coval_atom_idx = -1  # 实际上形成共价键的配体原子索引（从0开始）
                for atom_idx, atom_xyz in enumerate(ligand_pt['ligand_atom_coords']):
                    if round(atom_xyz[0].item(), 4) == round(float(coval_ligand_atom.coord[0]), 4):  # 保留前四位小数
                        coval_atom_idx = atom_idx
                        break
                if coval_atom_idx == -1:  # 未匹配到则跳过
                    continue
                for atom_idx, atom_xyz in enumerate(ligand_pt['ligand_atom_coords']):
                    for pdb_atom_xyz in ligand:
                        if round(atom_xyz[0].item(), 4) == round(float(pdb_atom_xyz.coord[0]), 4):
                            is_leaving_ligand[atom_idx] = 0

                ligand_covalent_bonds.append([
                    [item['chain_id2'], coval_res_atom, item['at_name1']],
                    ['ligand', coval_atom_idx], coval_res
                ])
        # 这种情况存在于组装体中，即只有一半含有配体
    for covalent_bond in copy.deepcopy(ligand_covalent_bonds):
        if covalent_bond[1][1] == -1:
            ligand_covalent_bonds.remove(covalent_bond)
        # 由于含有Object对象，因此只能迭代去重
    new_ligand_covalent_bond = []
    for bond in ligand_covalent_bonds:
        if len(new_ligand_covalent_bond) == 0:
            new_ligand_covalent_bond.append(bond)
        else:
            is_add = True
            for new_bond in new_ligand_covalent_bond:
                if new_bond[0][0] == bond[0][0] and new_bond[0][1].full_id[3][2] == new_bond[0][1].full_id[3][2]:
                    is_add = False
            if is_add:
                new_ligand_covalent_bond.append(bond)
    ligand_covalent_bonds = new_ligand_covalent_bond

    start_res_idx = 0
    for chain_i, chain in enumerate(select_structure[0]):
        residue_idx_list, coords_list, atom_mask, bfac_list, occ_list = get_amino_acid_coords(select_structure, chain_id=chain.id, max_atom_num=NTOTAL)

        res_seq = ''
        for residue in residue_idx_list:
            res_seq += seq1(residue.resname)

        holo_seq = protein_seq[chain_i]
        alter_coords_list = np.zeros((len(holo_seq), NTOTAL, 3))
        alter_bfac_list = np.full((len(holo_seq), NTOTAL), np.nan)
        alter_occ_list = np.zeros((len(holo_seq), NTOTAL))
        alter_atom_mask = np.zeros((len(holo_seq), NTOTAL))
        seq_mask = np.ones(len(holo_seq))
        aa_tokens = []

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -2
        # 让开 gap 很贵，避免为了解决首位不匹配而开一个 gap
        aligner.open_gap_score = -20
        aligner.extend_gap_score = -1
        aln = aligner.align(res_seq, holo_seq)[0]
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
            if not is_aa(seq3(holo_seq[i])):
                seq_mask[i] = 0.
                aa_tokens.append(20)
            else:
                aa_tokens.append(one_letter_token[holo_seq[i]])

        all_coords.append(alter_coords_list)
        all_bfac.append(alter_bfac_list)
        all_occ.append(alter_occ_list)
        all_atom_mask.append(alter_atom_mask)
        all_aa_token.extend(aa_tokens)
        all_seqs.append(str(holo_seq.seq))
        all_seq_mask.append(seq_mask)
        all_res_nb.append(np.arange(1 + res_num, len(holo_seq) + 1 + res_num))
        all_chains.extend([chain.id] * len(holo_seq))
        all_chain_idx.extend([chain_i] * len(holo_seq))

        res_num += (len(holo_seq) + RES_NB_JUMP)
        start_res_idx += len(aln_res_seq)

    xyz = torch.from_numpy(np.concatenate(all_coords)).float()

    # 设置口袋残基索引
    pocket_res_idx = []
    for pocket_res in pocket_res_list:
        try:
            pocket_idx, _, _ = find_nearest_atom(xyz, torch.from_numpy(pocket_res['CA'].coord))
        except:
            continue
        pocket_res_idx.append(pocket_idx)

    # 修正共价结合的蛋白质残基索引
    for ligand_covalent_bond in ligand_covalent_bonds:
        cova_res_idx, _, _ = find_nearest_atom(xyz, torch.from_numpy(ligand_covalent_bond[0][1].coord))
        ligand_covalent_bond[0][0] = all_chains[cova_res_idx]
        ligand_covalent_bond[0][1] = cova_res_idx  # 共价结合残基编号是基于0-base的
    if len(ligand_covalent_bonds) > 0:
        del ligand_covalent_bonds[-1]

    return {
            'holo_xyz': xyz,
            'holo_bfac': torch.from_numpy(np.concatenate(all_bfac)).float(),
            'holo_occ': torch.from_numpy(np.concatenate(all_occ)).float(),
            'holo_pocket_res': torch.tensor(pocket_res_idx),
            'holo_xyz_mask': torch.from_numpy(np.concatenate(all_atom_mask)),
            'holo_seq': all_seqs,
            'holo_aa_token': torch.tensor(all_aa_token),
            'holo_aa_mask': torch.from_numpy(np.concatenate(all_seq_mask)),
            'holo_icode': all_icode,
            'holo_chain': all_chains,
            'holo_chain_idx': torch.tensor(all_chain_idx),
            'holo_res_nb': torch.from_numpy(np.concatenate(all_res_nb)),
            'holo_ligand_covalent_bond': ligand_covalent_bonds,
            'holo_is_leaving_ligand': is_leaving_ligand,
    }
