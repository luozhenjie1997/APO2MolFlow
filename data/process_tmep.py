import pickle
import torch
import os
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict
from PDBBind_plus.utils import load_config, get_pdb_structure

"""
临时的分割代码，只取PDBBind+提供的部分，后面需要完善整个数据处理pipeline
"""

def find_nearest_atom(protein_coords, target_coord):
    """
    寻找张量中距离给定坐标最近的原子的索引
    """
    # 1. 利用广播机制直接相减，并计算最后一个维度（x,y,z）的 L2 范数（欧氏距离）
    distances = torch.norm(protein_coords - target_coord, dim=-1)

    # 找到整个 (L, 27) 矩阵中最小值的“展平”索引 (flattened index)
    min_flat_idx = torch.argmin(distances)

    # 将展平的索引还原回二维坐标 (残基索引, 原子索引)
    res_idx, atom_idx = torch.unravel_index(min_flat_idx, distances.shape)

    min_distance = distances[res_idx, atom_idx]

    return res_idx.item(), atom_idx.item(), min_distance.item()

def keep_tensor_and_list(pt_dict, keep):
    keep = keep.bool()
    for key, item in pt_dict.items():
        if type(item) == torch.Tensor and key not in ['ligand_is_leaving_ligand', 'pocket_res']:
            pt_dict[key] = item[keep]
        elif type(item) == list and len(item) == len(keep):
            new_item = []
            for flag, elem in zip(list(keep), item):
                if flag:
                    new_item.append(elem)
            pt_dict[key] = new_item
    return pt_dict

config, config_name = load_config('../config/base_config.yaml')

if __name__ == '__main__':
    apo_holo_dict = pickle.load(open(config['dataset']['base_path'] + '/use_single_apo_holo_dict.pkl', 'rb'))
    ligand_nums_dict = pickle.load(open(config['dataset']['base_path'] + '/ligand_nums_dict.pkl', 'rb'))
    only_use_single_ligand_list = pickle.load(open(config['dataset']['base_path'] + '/only_use_single_ligand_list.pkl', 'rb'))
    # plinder_df = pq.ParquetFile('/root/autodl-tmp/dataset/PLINDER/2024-06_v2_index_annotation_table.parquet').read().to_pandas()

    pbar = tqdm(range(len(apo_holo_dict)))

    no_single_chain = []

    for _, (holo_pdb_id, apo_pdb_list) in zip(pbar, apo_holo_dict.items()):
        if type(apo_pdb_list) is list:
            apo_id = apo_pdb_list[0].lower()
            apo_pt = torch.load(config['dataset']['apo_path'] + '/pt/%s/%s/%s.pt' % (apo_id[1:3], apo_id, apo_id))
            is_predict = False
        else:
            apo_pt = torch.load(config['dataset']['apo_path'] + '/candidate/af3/%s/%s/%s.pt' % (holo_pdb_id[1:3], holo_pdb_id, holo_pdb_id))
            is_predict = True
        ligand_num = ligand_nums_dict[holo_pdb_id]
        holo_pt = torch.load(config['dataset']['holo_path'] + '/%s/%s/%s.pt' % (holo_pdb_id[1:3], holo_pdb_id, holo_pdb_id))

        if type(holo_pt['ligand_is_leaving_ligand']) is dict:
            holo_pt['ligand_is_leaving_ligand'] = holo_pt['ligand_is_leaving_ligand'][0]
        if type(holo_pt['pocket_res']) is dict:
            holo_pt['pocket_res'] = holo_pt['pocket_res'][0]
        if type(holo_pt['ligand_covalent_bond']) is dict and len(holo_pt['ligand_covalent_bond']) > 0:
            holo_pt['ligand_covalent_bond'] = holo_pt['ligand_covalent_bond'][0]

        """为简单操作，仅涉及对称寡聚体"""
        if ligand_num is not None and ligand_num[1] > 1 and ligand_num[0] >= ligand_num[1]:
            # plinder_plis = plinder_df[plinder_df['entry_pdb_id'] == holo_pdb_id]['system_ligand_chains']

            ligand_pt = torch.load(config['dataset']['holo_path'] + '/%s/%s/%s_ligand.pt' % (holo_pdb_id[1:3], holo_pdb_id, holo_pdb_id))

            # 寻找该配体依赖的蛋白质亚基
            allow_chains = defaultdict(int)
            for atom in ligand_pt['ligand_coords'][0]:
                res_idx, _, rmsd = find_nearest_atom(holo_pt['xyz'], atom)
                allow_chains[holo_pt['chain'][res_idx]] += 1
            max_chain = max(allow_chains, key=allow_chains.get)

            # pocket_res = []
            # for res_idx in holo_pt['pocket_res']:
            #     if holo_pt['chain'][res_idx] == max_chain:
            #         pocket_res.append(res_idx.item())
            # holo_pt['pocket_res'] = torch.tensor(pocket_res)
            holo_pt['pocket_res'] = holo_pt['pocket_res']

            holo_pt['seq'] = [holo_pt['seq'][0]]

            keep = torch.tensor([True if chain == max_chain else False for chain in holo_pt['chain']])
            holo_pt = keep_tensor_and_list(holo_pt, keep)

            keep = torch.tensor([True if chain == apo_pt['chain'][0] else False for chain in apo_pt['chain']])
            apo_pt = keep_tensor_and_list(apo_pt, keep)
            apo_pt['seq'] = [apo_pt['seq'][0]]
            if is_predict:
                torch.save(apo_pt, config['dataset']['apo_path'] + '/candidate/af3/%s/%s/%s.pt' % (holo_pdb_id[1:3], holo_pdb_id, holo_pdb_id))
            else:
                torch.save(apo_pt, config['dataset']['apo_path'] + '/pt/%s/%s/%s.pt' % (apo_id[1:3], apo_id, apo_id))

        torch.save(holo_pt, config['dataset']['holo_path'] + '/%s/%s/%s.pt' % (holo_pdb_id[1:3], holo_pdb_id, holo_pdb_id))
