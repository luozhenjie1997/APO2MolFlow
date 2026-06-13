import pickle
import torch
import random
import gzip
import json
import model.utils.chemical as chemical
import model.utils.utils as utils
from torch.utils.data import Dataset
from Bio.Align import PairwiseAligner
from collections import defaultdict
from model.utils.chemical import NAATOKENS, num2aa, aa2long_noblank, aa2elt, aa2num, BBHeavyAtom
from model.utils.kinematics import xyz_to_t2d, get_chirals, xyz_to_c6d, c6d_to_bins, get_chiral_tags_from_mol
from model.utils.geometry import iterative_rigid_core_align, align_residue_level
from model.utils.util_module import XYZConverter_loader
from model.utils.geometry import construct_3d_basis, align, rot_matrix_to_quat

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # 只显示关键错误

"""
暂时不使用配体模板，后续看一下是否进行基于官能团的设计

预处理后的apo和holo会使用到的共有的信息（L_s表示所有链的残基之和）：
id：apo（holo）结构在rcsb上的id
xyz：[L_s, 37, 3]，坐标
xyz_mask：[L_s, 14]，坐标掩码
res_nb：[L_s]，残基编号（从1开始）
seq：[chain_nb, L]，完整序列
aa_token：[L_s]，完整序列对应的token
aa_mask：[L_s]，序列掩码
chain：[L_s]，每个残基所属的链Id

预处理的holo特有的信息（L_s表示所有链的残基之和）：
pocket_res：{chain: [L_p]}，其中[L_p]表示作为口袋的残基编号
ligand_name：配体三字符名称
ligand_is_leaving_ligand：[L_l]，标记配体中哪些原子为离去基团
ligand_covalent_bond：用于标记配体和蛋白质残基的共价连接信息。分两部分，第一部分表示共价连接的残基所在的链Id、残基编号（以pdb文件为准）以及形成共价连接的原子名称，
                      第二部分保存了和配体形成共价连接的原子索引（从0开始）
"""

class Apo2HoloDataset(Dataset):
    def __init__(self, base_path, apo_path, holo_path, holo_list_path, n_crop=384, crop_max_dist=20., atomize_protein=True,
                 crop_strategy='random_non_pocket', use_holo=False):
        super(Apo2HoloDataset, self).__init__()
        self.holo_ids = pickle.load(open(holo_list_path, 'rb'))  # 使用预保存的id列表，防止因为字典的无序性导致训练结果不一样
        self.base_path = base_path
        self.apo_path = apo_path
        self.holo_path = holo_path
        self.atomize_protein = atomize_protein  # 是否对共价连接的小配体-残基中的残基进行原子化
        self.xyz_converter = XYZConverter_loader()
        self.n_crop = n_crop
        self.crop_max_dist = crop_max_dist
        self.crop_strategy = crop_strategy
        self.use_holo = use_holo  # 是否将目标holo蛋白结构作为额外模板输入
        with gzip.open(base_path + '/ligands.json.gz', 'rt') as file:
            self.mols = json.load(file)  # 用于处理共价结合的蛋白质侧的残基

    def __len__(self):
        return len(self.holo_ids)

    def _keep_tensor_and_list(self, pt_dict, keep):
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

    def _keep_feature(self, feats_dict, keep):
        feats_dict['seq']['seq'] = feats_dict['seq']['seq'][keep]
        feats_dict['seq']['seq1hot'] = feats_dict['seq']['seq1hot'][keep]
        feats_dict['seq']['msa_latent'] = feats_dict['seq']['msa_latent'][:, keep]
        feats_dict['seq']['msa_full'] = feats_dict['seq']['msa_full'][:, keep]
        for key in feats_dict['xyzs'].keys():
            feats_dict['xyzs'][key] = feats_dict['xyzs'][key][keep]
        for key in feats_dict['template'].keys():
            if key == 't2d' or key == 'mask_t_2d':
                feats_dict['template'][key] = feats_dict['template'][key][:, keep][:, :,keep]
            else:
                feats_dict['template'][key] = feats_dict['template'][key][:, keep]
        feats_dict['bond_feats'] = feats_dict['bond_feats'][keep][:,keep]
        feats_dict['dist_matrix'] = feats_dict['dist_matrix'][keep][:,keep]
        feats_dict['idx'] = feats_dict['idx'][keep]
        feats_dict['same_chain'] = feats_dict['same_chain'][keep][:, keep]
        feats_dict['is_protein'] = feats_dict['is_protein'][keep]
        feats_dict['is_atomize_protein'] = feats_dict['is_atomize_protein'][keep]
        feats_dict['cova_mask'] = feats_dict['cova_mask'][keep]
        feats_dict['mask_protein_BB'] = feats_dict['mask_protein_BB'][keep]
        feats_dict['sm_mask'] = feats_dict['sm_mask'][keep]
        feats_dict['is_pocket_mask'] = feats_dict['is_pocket_mask'][keep]

        return feats_dict

    def __getitem__(self, idx):
        apo_holo_id = self.holo_ids[idx]
        # apo_holo_id = '2aay__A|1EPS__A'
        holo_id = apo_holo_id[:4]
        apo_id = apo_holo_id.split('|')[1][:4]
        processed_pt = torch.load(self.holo_path + '/%s/%s/%s.pt' % (holo_id[1:3], holo_id, apo_holo_id), weights_only=False)  # 加载预先进行部分处理的holo结构
        if apo_id == 'af3':
            is_predict = True
        else:
            is_predict = False

        holo_pt = {}
        apo_pt = {}
        holo_ligand_pt = {}
        for key, item in processed_pt.items():
            if 'holo_' in key:
                holo_pt[key.replace('holo_', '')] = item
            elif 'apo_' in key:
                apo_pt[key.replace('apo_', '')] = item
            elif 'ligand_' in key:
                holo_ligand_pt[key.replace('ligand_', '')] = item

        seq_is_eq = True
        for apo_seq, holo_seq in zip(apo_pt['seq'], holo_pt['seq']):
            if apo_seq != holo_seq:
                seq_is_eq = False
        if seq_is_eq:
            protein_xyz_mask = torch.logical_and(holo_pt['xyz_mask'].bool(), apo_pt['xyz_mask'])
            holo_core_mask = torch.ones(protein_xyz_mask.shape[0])
            alter_apo_holo_same_mask = holo_core_mask.clone()
        else:
            """目前有1.对齐后对非核心区域进行裁剪；2.对齐后进行填充，然后计算损失时忽略掉非核心区域。目前先尝试方案1"""
            # 标记增删位置的掩码
            apo_core_mask = torch.ones(apo_pt['aa_token'].shape[0], dtype=torch.int)
            holo_core_mask = torch.ones(holo_pt['aa_token'].shape[0], dtype=torch.int)
            # apo结构和holo的序列长度可能不一致，因此需要先找出多出来的部分
            total_apo_aln, total_holo_aln = '', ''
            for apo_seq, holo_seq in zip(apo_pt['seq'], holo_pt['seq']):
                aligner = PairwiseAligner()
                aligner.mode = "global"
                aligner.match_score = 2
                aligner.mismatch_score = -2
                # 让开 gap 很贵，避免为了解决首位不匹配而开一个 gap
                aligner.open_gap_score = -12
                aligner.extend_gap_score = -1
                aln = aligner.align(apo_seq, holo_seq)[0]
                apo_aln, holo_aln = aln[0], aln[1]  # aln_a/aln_b: 对齐后的字符串，'-' 表示 gap，长度相等
                total_apo_aln += apo_aln
                total_holo_aln += holo_aln
            apo_idx, holo_idx = 0, 0
            for apo, holo in zip(total_apo_aln, total_holo_aln):  # 标记缺失区域
                if apo == '-':
                    holo_core_mask[holo_idx] = False
                    holo_idx += 1
                elif holo == '-':
                    apo_core_mask[apo_idx] = False
                    apo_idx += 1
                else:
                    holo_idx += 1
                    apo_idx += 1
            # 将多出的部分删掉
            holo_pt = self._keep_tensor_and_list(holo_pt, holo_core_mask)
            apo_pt = self._keep_tensor_and_list(apo_pt, apo_core_mask)
            # 突变位置的掩码设置为False
            alter_apo_holo_same_mask = apo_pt['aa_token'] == holo_pt['aa_token']
            # 二者掩码取交集，并且将突变的位置的掩码设置为False
            protein_xyz_mask = torch.logical_and(holo_pt['xyz_mask'].bool(), apo_pt['xyz_mask']) * alter_apo_holo_same_mask[..., None]

        holo_length_dict = defaultdict(int)  # 链长度列表
        for chain in holo_pt['chain']:
            holo_length_dict[chain] += 1
        chain_lengths = [[chain, length] for chain, length in holo_length_dict.items()]
        term_info = utils.get_term_feats([item[1] for item in chain_lengths])
        chain_lengths.append(['ligand', holo_ligand_pt['atom_tokens'].shape[0]])  # 合并链信息

        # 需要按不同的链添加蛋白质连接特征
        protein_bond_feats_list = []
        for chain_length in chain_lengths:
            if chain_length[0] != 'ligand':
                protein_bond_feats_list.append(utils.get_protein_bond_feats(chain_length[1]))  # 蛋白质残基连接图，只有相邻的残基才会标记为5（残基-残基）
        protein_bond_feats = torch.zeros((holo_pt['res_nb'].shape[0], holo_pt['res_nb'].shape[0])).long()
        offset = 0
        for bf in protein_bond_feats_list:
            L = bf.shape[0]
            protein_bond_feats[offset:offset + L, offset:offset + L] = bf
            offset += L

        chain_bins = {}  # 链区间（0-base，左闭右开区间，可以直接作为切片索引）
        running_length = 0
        protein_len = len(holo_pt['aa_token'])
        for chain, length in chain_lengths:
            chain_bins[chain] = (running_length, running_length + length)
            running_length = running_length + length
        running_length -= chain_lengths[-1][1]  # 蛋白质的长度
        chains = list(chain_bins.keys())

        chain_idx = holo_pt['chain_idx']
        """
        原子化残基，仅当atomize_protein为True并且含有共价键时才进行操作。
        蛋白质氨基酸在进行共价结合时通常不会产生复杂的“离去基团”，主要变化是会失去一个质子（氢离子），而本项目不会对氢原子进行建模
        """
        cova_mask = torch.zeros_like(holo_pt['aa_mask'], dtype=torch.float)
        holo_sub_num = 0
        is_protein = []
        # 共价键信息，[[[蛋白质链ID, 形成共价键的残基索引（从1开始）, 形成共价键的原子名称], ['ligand', 形成共价键的原子索引（从0开始）]]]
        covalent_bonds = holo_pt['ligand_covalent_bond']
        cova_N_C_atom_index = []  # 记录形成共价结合的残基的N（氮）原子和C（碳）原子的索引，这两个原子分别连接上一个和下一个残基
        is_atomize_protein = [False] * holo_pt['aa_token'].shape[0]  # 用于标记哪个token为原子化后的残基
        if self.atomize_protein and len(covalent_bonds) > 0:
            cova_atom = []
            cova_index = []  # 记录形成共价键的原子索引
            start_idx = 0  # 用于对索引进行偏移
            real_bond_features = []
            holo_cova_xyzs = []
            apo_cova_xyzs = []
            cova_atom_mask = []
            for i, bond in enumerate(covalent_bonds):  # 开始原子化残基
                prot_chain, prot_res_idx, atom_to_bond = bond[0]
                sm_chid, sm_atom_num = bond[1]
                prot_res_idx = prot_res_idx - (~holo_core_mask[:prot_res_idx].bool()).sum()  # 共价结合的残基的实际位置（完整序列为基准，修正为从0开始）
                # 残基所包含的原子列表，只建模重原子
                res_atoms = aa2elt[holo_pt['aa_token'][prot_res_idx]][:14]
                res_atoms_name = aa2long_noblank[holo_pt['aa_token'][prot_res_idx]][:14]  # atom_to_bond是PDB的原子名称，因此需要进行索引定位
                cova_N_C_atom_index.append([0 + start_idx, 2 + start_idx])  # 这两个原子分别连接上一个和下一个残基
                cova_protein_atom = [a for a in res_atoms if a is not None]  # 共价结合的残基的原子名称
                cova_atom.extend(cova_protein_atom)
                chain_idx = torch.concat([chain_idx, torch.full((len(cova_protein_atom),), chains.index(prot_chain))])
                cova_index.append([start_idx + res_atoms_name.index(atom_to_bond), start_idx + len(cova_atom) + sm_atom_num])  # 记录形成共价键的原子索引
                residue_identity = num2aa[holo_pt['aa_token'][prot_res_idx]]  # 氨基酸三字母代码
                # 获取残基对应的sdf格式字串，并删掉一些原子（主要是OXT）
                residue_mol = self.mols[residue_identity]
                residue_mol_str = utils.delete_leaving_atoms_single_chain(residue_mol['sdf'], residue_mol['leaving'], is_str=True)
                # 获取原子化的残基的化学键信息（使用没有显式含芳香键的特征，防止后续因为芳香堆积导致的连接失败）
                atomize_real_bond_feats = utils.parse_mol(residue_mol_str, is_str=True)['real_bond_feats']
                real_bond_features.append(atomize_real_bond_feats)
                holo_cova_xyzs.append(holo_pt['xyz'][prot_res_idx][:len(cova_protein_atom)])
                apo_cova_xyzs.append(apo_pt['xyz'][prot_res_idx][:len(cova_protein_atom)])
                cova_atom_mask.append(holo_pt['xyz_mask'][prot_res_idx][:len(cova_protein_atom)])
                start_idx += len(cova_protein_atom)
                cova_mask[prot_res_idx] = 1.  # 标记发生共价结合的残基
            ligand_pair_is_leaving_ligand = ~holo_pt['ligand_is_leaving_ligand'][..., None] * ~holo_pt['ligand_is_leaving_ligand'][None, ...]
            # 需要去掉离去基团相应的特征。同样使用没有显式含芳香键的特征
            cova_ligand_atom = [num2aa[a] for a in holo_ligand_pt['atom_tokens'][~holo_pt['ligand_is_leaving_ligand']]]
            is_atomize_protein.extend([True] * len(cova_atom))
            cova_atom.extend(cova_ligand_atom)
            is_protein.extend([False] * len(cova_atom))
            is_atomize_protein.extend([False] * len(cova_ligand_atom))
            # 重新生成共价化合物时使用没有显示标记芳香键的特征
            real_bond_features.append(holo_ligand_pt['real_bond_feats'][ligand_pair_is_leaving_ligand].
                                 reshape((sum(~holo_pt['ligand_is_leaving_ligand']), sum(~holo_pt['ligand_is_leaving_ligand']))))
            holo_cova_xyzs.append(holo_ligand_pt['atom_coords'][~holo_pt['ligand_is_leaving_ligand']])
            apo_cova_xyzs.append(holo_ligand_pt['atom_coords'][~holo_pt['ligand_is_leaving_ligand']])
            cova_atom_mask.append(holo_ligand_pt['atom_mask'][~holo_pt['ligand_is_leaving_ligand']])
            cova_atom_token = torch.tensor([aa2num[atom] for atom in cova_atom])
            # 合并原子化的残基和配体的相关特征
            real_cova_bond_features, bond_feats, holo_cova_xyzs, cova_atom_mask = utils.get_combined_atoms_bonds(
                {'real_bond_feats_list': real_bond_features, 'xyzs': holo_cova_xyzs, 'cova_atom_mask': cova_atom_mask,
                 'bond_feats': holo_ligand_pt['bond_feats'][ligand_pair_is_leaving_ligand].reshape((sum(~holo_pt['ligand_is_leaving_ligand']), sum(~holo_pt['ligand_is_leaving_ligand'])))})
            apo_cova_xyzs = torch.concat(apo_cova_xyzs, dim=0)
             # 更新共价结合的立体化学特征
            conv_sdf_str, _, chirals, sm_bond_feats, sm_atom_frames, _ = utils.make_obmol_from_atoms_bonds_use_openbabel(
                cova_atom, holo_cova_xyzs, real_cova_bond_features, bond_feats, extra_bonds=cova_index, return_sdf=True)
            # chirals = get_chiral_tags_from_mol(conv_sdf_str)

            term_info = torch.concat([term_info, torch.zeros((cova_atom_token.shape[0], 2))], dim=0)  # N端/C端是多肽/蛋白质的特有概念
            seq = torch.concat([holo_pt['aa_token'], cova_atom_token], dim=-1)
            L_sum = seq.shape[0]
            bond_feats_list = [protein_bond_feats, sm_bond_feats]
            """
            将配体坐标扩展为(L_l, NTOTAL, 3)。
            由于RFAA核心是CA原子，因此将配体坐标放置在CA原子位置
            """
            holo_sm_xyzs = torch.concat([
                torch.zeros(holo_cova_xyzs.shape[0], 1, 3),
                holo_cova_xyzs.view(-1).reshape(holo_cova_xyzs.shape[0], 1, holo_cova_xyzs.shape[1]),
                torch.zeros(holo_cova_xyzs.shape[0], chemical.NTOTAL - 2, 3)], dim=1)  # 将CA原子位置设置为配体坐标
            apo_sm_xyzs = torch.concat([
                torch.zeros(apo_cova_xyzs.shape[0], 1, 3),
                apo_cova_xyzs.view(-1).reshape(apo_cova_xyzs.shape[0], 1, apo_cova_xyzs.shape[1]),
                torch.zeros(apo_cova_xyzs.shape[0], chemical.NTOTAL - 2, 3)], dim=1)
            sm_xyzs_mask = torch.concat([
                torch.zeros(cova_atom_mask.shape[0], 1),
                cova_atom_mask.reshape(-1, 1),
                torch.zeros(cova_atom_mask.shape[0], chemical.NTOTAL - 2)], dim=1)
        else:
            term_info = torch.concat([term_info, torch.zeros((holo_ligand_pt['atom_tokens'].shape[0], 2))], dim=0)  # N端/C端是多肽/蛋白质的特有概念
            seq = torch.concat([holo_pt['aa_token'], holo_ligand_pt['atom_tokens']], dim=-1)
            L_sum = seq.shape[0]
            bond_feats_list = [protein_bond_feats, holo_ligand_pt['bond_feats']]
            is_protein = [False] * len(holo_ligand_pt['atom_tokens'])
            is_atomize_protein.extend([False] * len(holo_ligand_pt['atom_tokens']))
            holo_sm_xyzs = holo_ligand_pt['atom_coords']
            holo_sm_xyzs = torch.concat([
                torch.zeros(holo_sm_xyzs.shape[0], 1, 3),
                holo_sm_xyzs.view(-1).reshape(holo_sm_xyzs.shape[0], 1, holo_sm_xyzs.shape[1]),
                torch.zeros(holo_sm_xyzs.shape[0], chemical.NTOTAL - 2, 3)], dim=1)  # 将配体坐标扩展为(L_l, NTOTAL, 3)
            apo_sm_xyzs = holo_sm_xyzs.clone()
            sm_xyzs_mask = holo_ligand_pt['atom_mask']
            sm_xyzs_mask = torch.concat([
                torch.zeros(sm_xyzs_mask.shape[0], 1),
                sm_xyzs_mask.reshape(-1, 1),
                torch.zeros(sm_xyzs_mask.shape[0], chemical.NTOTAL - 2)], dim=1)
            # smiles = holo_ligand_pt['ligand_SMILES']
            chirals = holo_ligand_pt['chirals']
            # chirals = get_chiral_tags_from_mol(holo_ligand_pt['ligand_sdf'])
            sm_atom_frames = holo_ligand_pt['atom_frames']

        cova_mask = torch.concat([cova_mask, torch.zeros(sm_xyzs_mask.shape[0])])
        # 用于标记对应的位置是否为蛋白质残基，对于原子化残基的原子也会标记为True
        is_protein = torch.tensor(([True] * len(holo_pt['aa_token'])) + is_protein)
        is_atomize_protein = torch.tensor(is_atomize_protein)  # 用于标记配体中哪些是原子化残基
        # chain_idx = torch.concat([chain_idx, torch.full(((~is_protein).sum() - is_atomize_protein.sum(),), holo_pt['chain_idx'][-1] + 1)])

        # 合并化学键类型特征
        bond_feats = torch.zeros((L_sum, L_sum)).long()
        offset = 0
        for bf in bond_feats_list:
            L = bf.shape[0]
            bond_feats[offset:offset + L, offset:offset + L] = bf
            offset += L
        msa_one_hot = torch.nn.functional.one_hot(seq[None], num_classes=NAATOKENS)  # MSA token部分
        # 对于本项目来说，由于MSA只剩下查询序列，因此可以直接合并
        msa_latent = torch.concat([msa_one_hot, msa_one_hot, torch.zeros(1, L_sum, 2), term_info[None]], dim=-1)  # 截断的MSA，tokens + tokens + 2 zeros + N/C
        msa_full = torch.concat([msa_one_hot, term_info[None]], dim=-1)  # 完整的MSA，tokens + N/C

        """
        初始化一维模板：
        1.token
        2.序列置信度，直接使用序列级别的掩码。配体部分的序列置信度直接设置为0
        3.结构置信度，配体部分直接置0，蛋白质部分使用CA原子的归一化温度因子（预测结构则使用残基级别的plddt）
        4.热点区域
        """
        t1d = torch.concat([msa_one_hot,
                            torch.concat([alter_apo_holo_same_mask[None, :, None], torch.zeros((1, holo_sm_xyzs.shape[0], 1))], dim=1),
                            torch.concat([torch.where(alter_apo_holo_same_mask.bool()[None, :, None], 0.,
                                                      torch.max(torch.full((alter_apo_holo_same_mask.shape[0],), 0.),
                                                                1 - (torch.nan_to_num(apo_pt['bfac'][:, 1], 100.) / 100))[None, :, None]),
                                          torch.zeros((1, holo_sm_xyzs.shape[0], 1))], dim=1),
                            torch.zeros((1, L_sum, 1))], dim=-1)

        pocket_res_nb = holo_pt['pocket_res']
        if not holo_core_mask.bool().all().item():
            pocket_res_nb = remap_indices(pocket_res_nb, holo_core_mask.bool())  # 将旧的蛋白质口袋残基索引映射至新的索引]
        # 根据口袋残基编号设置热点区域
        t1d[torch.arange(1), pocket_res_nb, -1] = 1.

        # holo_xyz_aligned, _, r, t = align(holo_pt['xyz'], apo_pt['xyz'], protein_xyz_mask)  # 将holo对齐到apo中
        # holo_sm_xyzs = holo_sm_xyzs @ r.T + t  # 配体移动至对齐后的位置
        # pocket_center = holo_xyz_aligned[pocket_res_nb][:, 1].mean(dim=0)  # 计算口袋区域中心（只涉及CA坐标）
        R, t_apo, t_holo, _ = iterative_rigid_core_align(apo_pt['xyz'][:, 1], holo_pt['xyz'][:, 1], protein_xyz_mask[:, :3].sum(dim=-1) == 3.0)
        apo_pt['xyz'] = torch.matmul((apo_pt['xyz'] - t_apo), R.T) + t_holo
        pocket_center = holo_pt['xyz'][pocket_res_nb][:, 1].mean(dim=0)  # 计算口袋区域中心（只涉及CA坐标）

        # 循环预测的坐标掩码
        xyzs_mask = torch.concat([protein_xyz_mask, sm_xyzs_mask], dim=0)
        # 循环预测的坐标（apo蛋白+配体（未插值）），并移动至口袋几何中心
        xyzs_prev = torch.concat([torch.concat([apo_pt['xyz'], torch.zeros(apo_pt['xyz'].shape[0], chemical.NTOTAL - apo_pt['xyz'].shape[1], 3)], dim=1), apo_sm_xyzs], dim=0) - pocket_center
        apo_pt['xyz'] = apo_pt['xyz'] - pocket_center
        # 真实目标坐标（holo蛋白+配体），并移动至口袋几何中心
        xyzs_true = torch.concat([torch.concat([holo_pt['xyz'], torch.zeros(holo_pt['xyz'].shape[0], chemical.NTOTAL - holo_pt['xyz'].shape[1], 3)], dim=1), holo_sm_xyzs], dim=0) - pocket_center
        # 获取真实坐标的替代坐标
        xyzs_true_alt = torch.zeros_like(xyzs_true)
        xyzs_true_alt.scatter_(1, chemical.long2alt[seq, :, None].repeat(1, 1, 3), xyzs_true)

        # 计算蛋白质刚体旋转矩阵和平移向量，真实结构需要做修正
        xyzs_prev_rots, xyzs_prev_trans = utils.rigid_from_3_points(xyzs_prev[:, BBHeavyAtom.N][None],
                                                                    xyzs_prev[:, BBHeavyAtom.CA][None],
                                                                    xyzs_prev[:, BBHeavyAtom.C][None], non_ideal=True)
        xyzs_true_rots, xyzs_true_trans = utils.rigid_from_3_points(xyzs_true[:, BBHeavyAtom.N][None],
                                                                    xyzs_true[:, BBHeavyAtom.CA][None],
                                                                    xyzs_true[:, BBHeavyAtom.C][None], non_ideal=True)
        xyzs_prev_rots, xyzs_prev_trans = xyzs_prev_rots[0], xyzs_prev_trans[0]
        xyzs_true_rots, xyzs_true_trans = xyzs_true_rots[0], xyzs_true_trans[0]
        # 计算蛋白质刚体旋转矩阵对应的四元数
        # xyzs_prev_rots_quat = rot_matrix_to_quat(xyzs_prev_rots)
        # xyzs_true_rots_quat = rot_matrix_to_quat(xyzs_true_rots)

        # 拼接蛋白质和配体编号
        input_idx = torch.concat([holo_pt['res_nb'], torch.arange(holo_pt['res_nb'][-1].item() + utils.RES_NB_JUMP + 1, holo_pt['res_nb'][-1].item() + utils.RES_NB_JUMP + 1 + holo_sm_xyzs.shape[0])])

        keep = torch.ones_like(seq).bool()  # 用于标记残基是否进行原子化
        if self.atomize_protein and len(covalent_bonds) > 0:
            # 连接原子化的残基
            for i, bond in enumerate(covalent_bonds):
                prot_chain, prot_res_idx, atom_to_bond = bond[0]
                prot_res_idx = prot_res_idx - holo_sub_num
                # 获取发生共价结合的蛋白质对应的链和配体对应的链的首尾索引（左闭右开区间）
                original_chain_start_index, original_chain_end_index = chain_bins[prot_chain]  # 被原子化的残基所在的链首尾索引

                # 原子化残基的N（氮）原子和C（碳）原子的索引，这两个原子分别连接上一个和下一个残基
                atomized_N_index = cova_N_C_atom_index[i][0]
                atomized_C_index = cova_N_C_atom_index[i][1]
                if prot_res_idx != original_chain_start_index:  # 若残基位于链首，则后续残基不形成额外键位
                    # 给前一残基与当前残基的N原子之间补上“残基-配体”原子键
                    bond_feats[prot_res_idx - 1, atomized_N_index + protein_len] = chemical.RESIDUE_ATOM_BOND  # bond_feats(L, L, 7)
                    bond_feats[atomized_N_index + protein_len, prot_res_idx - 1] = chemical.RESIDUE_ATOM_BOND
                if prot_res_idx != original_chain_end_index:  # 若残基位于链末端，则不与后续残基形成额外键
                    # 给当前残基的C原子与下一残基补上“残基-配体”原子键
                    bond_feats[prot_res_idx + 1, atomized_C_index + protein_len] = chemical.RESIDUE_ATOM_BOND  # bond_feats(L, L, 7)
                    bond_feats[atomized_C_index + protein_len, prot_res_idx + 1] = chemical.RESIDUE_ATOM_BOND
                keep[prot_res_idx] = False  # 原子化的残基在最终特征中被删除

        # TODO: 为了让模型以apo结构作为起点，因此这里不删除原子化残基的相关特征
        # # 删掉原子化残基的蛋白质中特征
        # seq, msa_latent, msa_full, bond_feats, xyzs_prev, xyzs_true, xyzs_prev_rots, xyzs_true_rots, xyzs_prev_trans, xyzs_true_trans, \
        # xyzs_mask, t1d, xyzs_t, xyzs_mask_t, term_info, idx, chirals, is_protein, is_atomize_protein = self._keep_feature({
        #     'seq': seq, 'msa_latent': msa_latent, 'msa_full': msa_full, 'bond_feats': bond_feats, 'xyzs_true': xyzs_true,
        #     'xyzs_prev_rots': xyzs_prev_rots, 'xyzs_true_rots': xyzs_true_rots, 'xyzs_prev_trans': xyzs_prev_trans,
        #     'xyzs_true_trans': xyzs_true_trans, 'xyzs_prev': xyzs_prev, 'xyzs_mask': xyzs_mask, 't1d': t1d, 'xyzs_t': xyzs_t,
        #     'xyzs_mask_t': xyzs_mask_t, 'term_info': term_info, 'idx': input_idx, 'chirals': chirals, 'is_protein': is_protein,
        #     'is_atomize_protein': is_atomize_protein
        # }, keep)
        L_sum = seq.shape[0]

        dist_matrix = utils.get_bond_distances(bond_feats)  # 距离矩阵
        same_chain = utils.same_chain_from_bond_feats(bond_feats)  # 获取是否属于同一个链的pairwise掩码

        """
        构造模板：
        1. 默认只使用apo模板，表示模型的结构起点；
        2. 当use_holo=True时，额外加入目标holo蛋白模板，表示希望配体诱导到达的蛋白构象。
        注意目标holo模板只暴露蛋白坐标，配体部分保持零坐标和无效mask，避免泄漏真实配体信息。
        """
        apo_template_xyz = torch.concat([
            apo_pt['xyz'],
            torch.zeros(apo_pt['xyz'].shape[0], chemical.NTOTAL - apo_pt['xyz'].shape[1], 3, device=apo_pt['xyz'].device)
        ], dim=1)
        apo_template_mask = torch.concat([
            apo_pt['xyz_mask'],
            torch.zeros(apo_pt['xyz_mask'].shape[0], chemical.NTOTAL - apo_pt['xyz_mask'].shape[1], device=apo_pt['xyz_mask'].device)
        ], dim=1).bool()

        zero_sm_xyzs = torch.zeros_like(holo_sm_xyzs)
        zero_sm_mask = torch.zeros_like(sm_xyzs_mask)
        template_xyzs = [torch.concat([apo_template_xyz, zero_sm_xyzs], dim=0)]
        template_masks = [torch.concat([apo_template_mask, zero_sm_mask], dim=0)]
        template_t1d = [t1d[0]]

        if self.use_holo:
            target_holo_template_xyz = torch.concat([
                holo_pt['xyz'] - pocket_center,
                torch.zeros(holo_pt['xyz'].shape[0], chemical.NTOTAL - holo_pt['xyz'].shape[1], 3, device=holo_pt['xyz'].device)
            ], dim=1)
            target_holo_template_mask = torch.concat([
                holo_pt['xyz_mask'],
                torch.zeros(holo_pt['xyz_mask'].shape[0], chemical.NTOTAL - holo_pt['xyz_mask'].shape[1], device=holo_pt['xyz_mask'].device)
            ], dim=1).bool()
            target_holo_t1d = t1d[0].clone()
            target_holo_t1d[~is_protein, :] = 0.  # 目标holo模板不提供任何配体一维信息
            template_xyzs.append(torch.concat([target_holo_template_xyz, zero_sm_xyzs], dim=0))
            template_masks.append(torch.concat([target_holo_template_mask, zero_sm_mask], dim=0))
            template_t1d.append(target_holo_t1d)

        xyzs_t = torch.stack(template_xyzs, dim=0)
        xyzs_mask_t = torch.stack(template_masks, dim=0).bool()
        t1d = torch.stack(template_t1d, dim=0)
        mask_t_2d = utils.get_prot_sm_mask(xyzs_mask_t, seq)  # 指出模板结构的有效区域
        mask_t_2d = mask_t_2d[:, None] * mask_t_2d[:, :, None]  # pairwise掩码
        mask_t_2d = mask_t_2d * same_chain.bool()[None]  # 忽略掉不同链间的对
        xyz_t_frame = utils.xyz_t_to_frame_xyz(xyzs_t[None], seq, sm_atom_frames)[0]  # 将模板中的小分子配体的坐标映射到局部坐标系
        t2d = xyz_to_t2d(xyz_t_frame[None], mask_t_2d[None])[0]  # 二维模板
        template_seq = seq[None].expand(xyzs_t.shape[0], -1)
        alpha_t, _, alpha_t_mask, _ = self.xyz_converter.get_torsions(xyzs_t.reshape(-1, L_sum, chemical.NTOTAL, 3), template_seq,
                                                                      mask_in=xyzs_mask_t.reshape(-1, L_sum, chemical.NTOTAL))  # 获取模板扭转角和扭转角掩码
        torsion_angle_prev, _, alpha_prev_mask, _ = self.xyz_converter.get_torsions(xyzs_prev[None], seq[None], mask_in=xyzs_mask.reshape(-1, L_sum, chemical.NTOTAL))  # 获取apo结构扭转角和扭转角掩码
        torsion_angle_true, torsion_angle_alt_true, alpha_true_mask, planar = self.xyz_converter.get_torsions(xyzs_true[None], seq[None], mask_in=xyzs_mask.reshape(-1, L_sum, chemical.NTOTAL))  # 获取holo结构扭转角和扭转角掩码
        alpha_mask = alpha_prev_mask * alpha_true_mask
        # 将掩码作为模板扭转角特征
        alpha_t = torch.concat([alpha_t, alpha_t_mask.reshape(xyzs_t.shape[0], L_sum, chemical.NTOTALDOFS, 1)], dim=-1)
        mask_protein_BB = ~(xyzs_mask[:, :3].sum(dim=-1) < 3.0)  # 根据蛋白质主链情况来确定是否计算损失
        # 该掩码用于计算口袋区域的相关损失（包括配体）
        is_pocket_mask = torch.where(is_protein, False, True)
        is_pocket_mask[pocket_res_nb] = True

        data_dict = {'seq': {'seq': seq, 'seq1hot': msa_full[0, :, :NAATOKENS], 'msa_latent': msa_latent, 'msa_full': msa_full,},
                     'xyzs': {'xyzs_true': xyzs_true, 'xyzs_true_rots': xyzs_true_rots, 'xyzs_prev_trans': xyzs_prev_trans,
                              'xyzs_prev': xyzs_prev, 'xyzs_prev_rots': xyzs_prev_rots, 'xyzs_true_trans': xyzs_true_trans,
                              'xyzs_mask': xyzs_mask.bool(), 'torsion_angle_prev': torsion_angle_prev[0], 'torsion_angle_true': torsion_angle_true[0],
                              'torsion_angle_alt_true': torsion_angle_alt_true[0], 'planar': planar[0], 'torsion_mask': alpha_mask[0],
                              'res_mask': (xyzs_mask[:, 1] == 1) * is_protein, 'xyzs_true_alt': xyzs_true_alt},
                     'template': {'xyz_t': xyzs_t, 't1d': t1d, 't2d': t2d, 'alpha_t': alpha_t, 'mask_t': xyzs_mask_t,
                                  'mask_t_2d': mask_t_2d, 'alpha_t_mask': alpha_t_mask, },
                     'bond_feats': bond_feats, 'dist_matrix': dist_matrix, 'chirals': chirals.int(),
                     'atom_frames': sm_atom_frames, 'idx': input_idx, 'same_chain': same_chain, 'is_protein': is_protein,
                     'is_atomize_protein': is_atomize_protein, 'cova_mask': cova_mask, 'is_pocket_mask': is_pocket_mask,
                     'mask_protein_BB': mask_protein_BB, 'sm_mask': ~is_protein, 'is_predict': is_predict,
                     'apo_pdb_id': apo_id, 'holo_pdb_id': holo_id}

        if seq.shape[0] > self.n_crop:
            # keep_mask = pocket_based_crop(holo_pt['aa_token'].shape[0], holo_pt['pocket_res'], chain_lengths, self.n_crop - sm_xyzs_mask.shape[0])
            max_protein_residues = self.n_crop - sm_xyzs_mask.shape[0]
            if self.crop_strategy == 'random_non_pocket':
                # 口袋残基全部保留，非口袋残基随机补足
                keep_mask = pocket_random_crop(xyzs_true[is_protein, 1].shape[0], pocket_res_nb, max_residues=max_protein_residues)
            elif self.crop_strategy == 'distance':
                pocket_center = xyzs_true[pocket_res_nb, 1].mean(dim=0)
                keep_mask = pocket_CA_crop(xyzs_true[is_protein, 1], pocket_center, max_residues=max_protein_residues,
                                           radius=self.crop_max_dist)  # 获取保留残基的掩码
            else:
                raise ValueError("crop_strategy must be 'random_non_pocket' or 'distance'")
            keep_mask = torch.concat([keep_mask, torch.ones(sm_xyzs_mask.shape[0], dtype=torch.bool, device=keep_mask.device)])  # 配体部分全部保留
            data_dict = self._keep_feature(data_dict, keep_mask)

        c6d = xyz_to_c6d(data_dict['xyzs']['xyzs_true'][None])
        data_dict['xyzs']['c6d'] = c6d_to_bins(c6d, same_chain=same_chain)[0]

        frames, frame_mask = utils.get_frames(data_dict['xyzs']['xyzs_mask'], data_dict['seq']['seq'], chemical.frame_indices,
                                              data_dict['atom_frames'])
        data_dict['frames'] = frames
        data_dict['frame_mask'] = frame_mask

        return data_dict


def remap_indices(pocket_indices, keep_mask):
    """
    将原始索引映射到删除残基后的新索引。
    """
    # 筛选出那些在 keep_mask 中为 True 的口袋残基
    valid_pocket_mask = keep_mask[pocket_indices]
    valid_indices = pocket_indices[valid_pocket_mask]

    if len(valid_indices) == 0:
        return torch.tensor([], dtype=torch.long, device=pocket_indices.device)

    # 构建映射表
    # 例子: keep_mask = [1, 0, 1, 1]
    # cumsum        = [1, 1, 2, 3]
    # mapping       = [0, -1, 1, 2] (对应保留下来的 0, 1, 2 号位)

    # 计算累积和，得到每个位置是第几个"保留残基"
    cumsum = torch.cumsum(keep_mask.long(), dim=0)

    # 构建全映射表: old_idx -> new_idx。索引从 0 开始，所以要减 1
    mapping_table = cumsum - 1

    new_pocket_indices = mapping_table[valid_indices]  # 查表获取新索引

    return new_pocket_indices


def pocket_CA_crop(protein_ca_xyz, pocket_center, radius=20.0, max_residues=384, k_expand=2):
    """
    基于口袋中心进行蛋白质裁剪。
    :param protein_ca_xyz: (N_res, 3) 蛋白质 CA 坐标
    :param pocket_center: (3,) 口袋中心坐标
    :param radius: 裁剪半径 (埃)
    :param max_residues: 最大保留残基数
    :return: (N_res,) 裁剪掩码
    """

    num_res = protein_ca_xyz.shape[0]

    # 距离筛选
    dists = torch.linalg.norm(protein_ca_xyz - pocket_center, dim=-1)
    mask = dists < radius

    # 使用卷积进行序列扩展
    mask_float = mask.float().view(1, 1, -1)  # (1, 1, N)

    # 全1卷积核，大小为 2*k + 1。这相当于对每个选中点，向左右各"涂抹"k个位置
    kernel_size = 2 * k_expand + 1
    padding = k_expand
    kernel = torch.ones(1, 1, kernel_size, device=mask.device)
    expanded = torch.nn.functional.conv1d(mask_float, kernel, padding=padding)  # 执行 1D 卷积
    expanded_mask = expanded.view(-1) > 0

    # 防止padding导致的越界。
    expanded_mask = expanded_mask[:num_res]

    current_selected = expanded_mask.sum().item()

    if current_selected > max_residues:
        # 在所有被 expanded_mask 覆盖的残基中，只保留距离最近的max_residues个。
        valid_dists = torch.where(
            expanded_mask,
            dists,
            torch.tensor(float('inf'), device=dists.device)
        )

        # 选出距离最小的 max_residues 个索引
        _, topk_indices = torch.topk(valid_dists, k=max_residues, largest=False)

        # 重置掩码
        expanded_mask = torch.zeros_like(expanded_mask)
        expanded_mask[topk_indices] = True

        return expanded_mask
    elif current_selected < max_residues:
        num_to_add = max_residues - current_selected
        # 将已经被选中的残基距离设为无穷大，这样它们就不会被重复选中
        unselected_dists = torch.where(~expanded_mask, dists, torch.tensor(float('inf'), device=dists.device))

        # 选出剩余未选中残基中，距离口袋中心最小的num_to_add个索引
        _, topk_indices = torch.topk(unselected_dists, k=num_to_add, largest=False)

        # 将这些被补充的残基在掩码中设为 True
        expanded_mask[topk_indices] = True

        return expanded_mask

    return expanded_mask


def pocket_random_crop(num_res, pocket_indices, max_residues=384):
    """
    保留全部口袋残基，并从非口袋残基中随机补足到目标数量。
    :param num_res: 蛋白质残基总数
    :param pocket_indices: 口袋残基索引
    :param max_residues: 目标保留的蛋白质残基数量
    :return: (num_res,) 裁剪掩码
    """
    keep_mask = torch.zeros(num_res, dtype=torch.bool, device=pocket_indices.device)
    if num_res == 0:
        return keep_mask

    valid_pocket_indices = pocket_indices[(pocket_indices >= 0) & (pocket_indices < num_res)].long()
    keep_mask[valid_pocket_indices] = True

    current_selected = keep_mask.sum().item()
    if current_selected >= max_residues:
        return keep_mask

    num_to_add = min(max_residues - current_selected, num_res - current_selected)
    if num_to_add <= 0:
        return keep_mask

    non_pocket_indices = torch.nonzero(~keep_mask, as_tuple=False).reshape(-1)
    random_indices = torch.randperm(non_pocket_indices.shape[0], device=non_pocket_indices.device)[:num_to_add]
    keep_mask[non_pocket_indices[random_indices]] = True

    return keep_mask


if __name__ == '__main__':
    random.seed(42)
    dataset = Apo2HoloDataset(base_path='/root/autodl-tmp/dataset/APO2MolFlow', apo_path='/root/autodl-tmp/dataset/APO2MolFlow/apo',
                              holo_path='/root/autodl-tmp/dataset/APO2MolFlow/holo',
                              holo_list_path='/root/autodl-tmp/dataset/APO2MolFlow/use_holo_ids_filter_mw.pkl', n_crop=384, atomize_protein=True)
    """
    部分测试实例：
    4622(5t9w)  不含共价键（单链，apo部分对齐）
    0(10gs)     不含共价键（多链，apo部分对齐）
    5542(6p3a)  不含共价键（多链，无需对齐，界面结合）
    
    30(4yva)    含有共价键（单链，无需对齐，无离去基团）
    6794(8ao3)  含有共价键（单链，无需对齐，有离去基团）
    1979(3ika)  含有共价键（多链，holo部分对齐，无离去基团）
    6419(7or4)  含有共价键（多链，无需对齐，有离去基团）
    
    6563(7rvo)  口袋首尾相连后过长
    6442(7pop)  超长蛋白
    
    2315        中间有缺失残基以及突变残基
    4215(5hh5)  
    """
    a = dataset[72]
    # 自检确保所有数据的处理均没有问题
    for i in range(len(dataset)):
        a = dataset[i]
        print(i)
