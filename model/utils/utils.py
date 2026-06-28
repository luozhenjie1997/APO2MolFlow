import os
import re
import torch
import scipy
import networkx as nx
import model.utils.chemical as chemical
import yaml
import random
import math
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from easydict import EasyDict
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
from openbabel import pybel
from model.utils.parsers import clean_sdffile
from model.utils.kinematics import get_chirals

PocketMiner_train_apo_holo_ids = (('3PPN', '3PPR'), ('4W51', '4W58'), ('4I92', '4I94'), ('6E5D', '6E5F'), ('5H9A', '6E5L'),
                                  ('4IC4', '4INQ'), ('1EZM', '3DBK'), ('1S2O', '1U2S'), ('1TVQ', '1TW4'), ('2OY4', '3DPF'),
                                  ('5NIA', '5NI6'), ('1Y1A', '1Y1A'), ('3NX1', '3NX2'), ('1KX9', '1N8V'), ('5NZM', '2OEG'),
                                  ('6YPK', '5OSZ'), ('2CEY', '6H76'), ('1URP', '2DRI'), ('3FVJ', '2B03'), ('3UGK', '3UH1'),
                                  ('2ZKU', '3VQS'), ('5ZA4', '5YYB'), ('4V38', '4V3B'), ('3P53', '6I11'), ('2HQ8', '2HPS'),
                                  ('2W9T', '2W9S'), ('4R72', '4R74'), ('3QXW', '3QXV'), ('3KJE', '3KJG'), ('2LAO', '1LAH'),
                                  ('4P0I', '5OTA'), ('1KMO', '1KMP'), ('1J8F', '6QCN'), ('5UXA', '5IGY'), ('6HB0', '6HBD'),
                                  ('2FJY', '2P70'), ('5G1M', '5G3R'), ('6RVM', '5XDT'))

rfaa_num2aa=[
        'ALA','ARG','ASN','ASP','CYS',
        'GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO',
        'SER','THR','TRP','TYR','VAL',
        'UNK','MAS',
        ' DA',' DC',' DG',' DT', ' DX',
        ' RA',' RC',' RG',' RU', ' RX',
        'HIS_D', # 组氨酸的一种特定质子化状态。仅用于cart_bonded
        'Al', 'As', 'Au', 'B',
        'Be', 'Br', 'C', 'Ca', 'Cl',
        'Co', 'Cr', 'Cu', 'F', 'Fe',
        'Hg', 'I', 'Ir', 'K', 'Li', 'Mg',
        'Mn', 'Mo', 'N', 'Ni', 'O',
        'Os', 'P', 'Pb', 'Pd', 'Pr',
        'Pt', 'Re', 'Rh', 'Ru', 'S',
        'Sb', 'Se', 'Si', 'Sn', 'Tb',
        'Te', 'U', 'W', 'V', 'Y', 'Zn',
        'ATM'
    ]

# RDKit显式区分单/双/三/芳香键
rdkit_bond_type_lookup = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC
}

RES_NB_JUMP = 50  # 链间索引跳号


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

# 设置随机种子
def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}


def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]


def load_rfaa_weights_without_nucleic_acids(model, rfaa_ckpt_path, device='cpu', total_rfaa_main_blocks=32, my_main_blocks=16):
    """
    加载 RFAA 权重，自动处理因剔除核酸 token 导致的维度不匹配问题。
    """
    checkpoint = torch.load(rfaa_ckpt_path, map_location=device, weights_only=False)
    rfaa_state_dict = checkpoint.get('model_state_dict', checkpoint)

    my_model_dict = model.state_dict()

    # 构建新的、过滤后的权重字典
    pretrained_dict = {}

    # --- 核心：构建索引映射 ---
    mapping = []
    unmapped_tokens = []
    for i, token in enumerate(chemical.num2aa):
        if token in rfaa_num2aa:
            mapping.append(rfaa_num2aa.index(token))
        else:
            mapping.append(-1)  # 在 RFAA 中找不到，标记为 -1
            unmapped_tokens.append(token)

    # RFAA 的词表大小
    rfaa_vocab_size = len(rfaa_num2aa)
    my_vocab_size = len(chemical.num2aa)

    # --- 计算层数偏移量 ---
    # 如果 RFAA 有 36 层，我们保留 12 层，则 offset = 24。
    # RFAA 的第 24 层将映射到我们的第 0 层。
    offset = total_rfaa_main_blocks - my_main_blocks
    mapped_layers_count = 0

    # --- 开始遍历权重并智能映射键名 ---
    for rfaa_key, rfaa_weight in rfaa_state_dict.items():
        mapped_key = rfaa_key

        # 1. 拦截并处理主干网络 (Iterative Blocks) 的层索引
        # 使用正则匹配类似 'block.24.' 或 'trunk.block.24.' 的结构
        match = re.search(r"(simulator\.main_block\.)(\d+)(\..*)", rfaa_key)
        if match:
            prefix = match.group(1)  # 例如 "trunk.blocks."
            layer_idx = int(match.group(2))  # 例如 24
            suffix = match.group(3)  # 例如 ".attn.linear.weight"

            if layer_idx < offset:
                # 丢弃属于前 total_rfaa_blocks - my_blocks 层的权重
                continue
            else:
                # 重新映射索引：将 24 映射为 0，25 映射为 1...
                new_idx = layer_idx - offset
                mapped_key = f"{prefix}{new_idx}{suffix}"
                mapped_layers_count += 1

        # 如果映射后的键不在我们的模型中，跳过 (例如结构精修层以外的多余头)
        if mapped_key not in my_model_dict:
            continue

        my_weight = my_model_dict[mapped_key]

        # 2. 形状对齐与词汇表维度修正 (应用到映射后的 mapped_key)
        if rfaa_weight.shape == my_weight.shape:
            pretrained_dict[mapped_key] = rfaa_weight
        else:
            dim_to_map = -1
            if rfaa_weight.shape[0] == rfaa_vocab_size and my_weight.shape[0] == my_vocab_size:
                dim_to_map = 0
            elif rfaa_weight.shape[-1] == rfaa_vocab_size and my_weight.shape[-1] == my_vocab_size:
                dim_to_map = rfaa_weight.dim() - 1

            if dim_to_map != -1:
                adapted_weight = my_weight.clone()
                for my_idx, rfaa_idx in enumerate(mapping):
                    if rfaa_idx != -1:
                        if dim_to_map == 0:
                            adapted_weight[my_idx] = rfaa_weight[rfaa_idx]
                        else:
                            adapted_weight[..., my_idx] = rfaa_weight[..., rfaa_idx]
                pretrained_dict[mapped_key] = adapted_weight
            else:
                pretrained_dict[mapped_key] = my_weight

    # 更新模型
    my_model_dict.update(pretrained_dict)
    model.load_state_dict(my_model_dict, strict=False)
    print("RFAA weights successfully adapted and loaded!")
    return model


def is_protein(seq):
    return seq < chemical.NPROTAAS

def get_automorphs(mol, xyz_sm, mask_sm, max_symm=1000):
    """
    枚举原子的对称排列，为小分子生成所有原子对称排列后的坐标与掩码，用于后续推理时考虑配体的对称性
    """
    try:
        automorphs = openbabel.vvpairUIntUInt()
        openbabel.FindAutomorphisms(mol, automorphs)  # 找出所有在化学属性上完全等价的原子排列方式

        # 每行形如(i, j)，每一行代表一种对称变换方案，记录了“原原子索引”到“对称后原子索引”的映射关系
        automorphs = torch.tensor(automorphs)
        n_symmetry = automorphs.shape[0]

        xyz_sm = xyz_sm[None].repeat(n_symmetry,1,1)
        mask_sm = mask_sm[None].repeat(n_symmetry,1)

        # 按同构映射重新排列每个对称解的坐标和掩码，实现原子交换
        xyz_sm = torch.scatter(xyz_sm, 1, automorphs[:,:,0:1].repeat(1,1,3),
                               torch.gather(xyz_sm,1,automorphs[:,:,1:2].repeat(1,1,3)))
        mask_sm = torch.scatter(mask_sm, 1, automorphs[:,:,0],
                            torch.gather(mask_sm, 1, automorphs[:,:,1]))
    except Exception as e:  # 若枚举失败，返回原始单份坐标和掩码
        xyz_sm = xyz_sm[None]
        mask_sm = mask_sm[None]
    if xyz_sm.shape[0] > max_symm:  # 避免输出过大
        xyz_sm = xyz_sm[:max_symm]
        mask_sm = mask_sm[:max_symm]
    return xyz_sm, mask_sm

### 生成小分子键特征 ###
def get_bond_feats(mol):
    """为小分子创建二维键图"""
    N = mol.NumAtoms()
    bond_feats = torch.zeros((N, N)).long()
    real_bond_feats = torch.zeros((N, N)).long()

    for bond in openbabel.OBMolBondIter(mol):
        i,j = (bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1)
        bond_feats[i, j] = bond.GetBondOrder() if not bond.IsAromatic() else 4  # 4表示芳香键
        bond_feats[j, i] = bond_feats[i,j]
        real_bond_feats[i, j] = bond.GetBondOrder()
        real_bond_feats[j, i] = real_bond_feats[i, j]

    return bond_feats.long(), real_bond_feats.long()


### 创建原子框架用于FAPE损失计算 ###
def get_nxgraph(mol):
    '''从openbabel的OBMol构建NetworkX图'''

    N = mol.NumAtoms()

    # 遍历所有化学键组合键合原子对，OpenBabel索引从1开始，请重新调整为从0开始的索引
    bonds = [(bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1) for bond in openbabel.OBMolBondIter(mol)]

    # 连接图
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)

    return G


def find_all_rigid_groups(bond_feats):
    """
    remove all single bonds from the graph and find connected components
    """
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    rigid_atom_bonds_np = rigid_atom_bonds[0].cpu().numpy()
    G = nx.from_numpy_array(rigid_atom_bonds_np)
    connected_components = nx.connected_components(G)
    connected_components = [cc for cc in connected_components if len(cc)>2]
    connected_components = [torch.tensor(list(combinations(cc,2))) for cc in connected_components]
    if connected_components:
        connected_components = torch.cat(connected_components, dim=0)
    else:
        connected_components = None
    return connected_components


def find_all_paths_of_length_n(G : nx.Graph, n : int, **karg) -> torch.Tensor:
    '''在networkx图中查找所有长度为N的路径
    https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph'''

    # 采用深度优先搜索递归策略
    def findPaths(G, u, n):
        if n==0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1)
                 if u not in path]  # 确保路径中不会出现重复节点
        return paths

    # all paths of length n
    allpaths = [tuple(p) if p[0]<p[-1] else tuple(reversed(p))  # 比较路径起点和终点的索引大小，始终保持索引较小的一端作为起点
                for node in G for p in findPaths(G,node,n)]

    # 选择是否保留所有排列
    if 'omit_permutation' in karg.keys() and not karg['omit_permutation']:
        allpaths = [tuple(p) for node in G for p in findPaths(G,node,n)]

    # 确保最终结果中每条无向路径只出现一次
    allpaths = list(set(allpaths))

    #return torch.tensor(allpaths)
    return allpaths

def get_atom_frames(msa, G, **karg):
    """
    为分子中的每个原子选择由3个键合原子组成的骨架，基于规则的系统根据原子优先级选择骨架
    """
    query_seq = msa
    frames = find_all_paths_of_length_n(G, 2, **karg)  # 得到所有长度为2的路径。这些路径表示三连原子（两条键顺连），作为候选三元组
    selected_frames = []
    # 遍历分子中的每一个原子n，尝试为其分配最合适的框架
    for n in range(msa.shape[0]):
        frames_with_n = [frame for frame in frames if n == frame[1]]  # 首选路径的中间节点为n的框架（b == n），保证它是中心原子

        # 某些原子（如末端的氢原子或卤素原子）可能只有一个邻居，无法作为中心点构成路径。此时会退而求其次，寻找任何包含原子n的路径（n可能在路径的末端）
        if not frames_with_n:
            frames_with_n = [frame for frame in frames if n in frame]
        # 若原子完全孤立（不在任何原子框架内），则设置为占位符，并且在损失计算中忽略该原子
        if not frames_with_n:
            selected_frames.append([(0, 1), (0, 1), (0, 1)])
            continue
        frame_priorities = []
        for frame in frames_with_n:  # 为了保证模型的确定性（即同一分子每次运行生成的坐标系必须一致），引入了优先级系统
            # 虽然有些粗糙，但它使用“query_seq”将原子的索引转换为“原子类型”，再将其转换为优先级。
            indices = [index for index in frame if index != n]
            # 通过num2aa映射回原子类型，再用atom2frame_priority得到优先级整数
            aas = [chemical.num2aa[int(query_seq[index].numpy())] for index in indices]
            if 'omit_permutation' in karg.keys() and not karg['omit_permutation']:
                # 未显式设置 omit_permutation=True，保留次序构建[priority1, priority2]
                frame_priorities.append([chemical.atom2frame_priority[aa] for aa in aas])
            else:
                # 对两个优先级排序后比较，使得框架只按组成集合而非顺序区分
                frame_priorities.append(sorted([chemical.atom2frame_priority[aa] for aa in aas]))

        # np.argsort无法正确排序元组，因此只需使用键对索引列表进行排序即可。
        sorted_indices = sorted(range(len(frame_priorities)), key=lambda i: frame_priorities[i])  # 选取最小的优先级作为最佳框架
        # 计算框架的相对偏移量。例如，如果当前原子索引是10，选定的参考框架索引是[9, 10, 11]，则存储为偏移量[-1, 0, 1]
        frame = [(frame - n, 1) for frame in frames_with_n[sorted_indices[0]]]
        selected_frames.append(frame)
    assert msa.shape[0] == len(selected_frames)  # [num_atom, 3, 2]，其中包含了每个原子构建坐标轴所需的参考原子相对于自己的位置和掩码（固定为1）
    return torch.tensor(selected_frames).long()

def is_atom(seq):
    """判断一个token列表中的每个token是否为元素类型"""
    return seq > chemical.NNAPROTAAS

def get_term_feats(Ls):
    """
    创建N/C端二进制特征。返回[L,2]，其中只有N端和C端的token为1，其余均为0
    Ls:  复合物长度列表
    """
    term_info = torch.zeros((sum(Ls),2)).float()
    start = 0
    for L_chain in Ls:
        term_info[start, 0] = 1.0  # N端标记
        term_info[start+L_chain-1,1] = 1.0  # C端标记
        start += L_chain
    return term_info

def get_bond_distances(bond_feats):
    atom_bonds = (bond_feats > 0) * (bond_feats<5)  # 排除没有键链接以及其余特殊的键
    # 使用图搜索计算最短路径。directed=False表示视作无向图
    dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds.long().numpy(), directed=False)
    # dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0)) # protein portion is inf and you don't want to mask it out
    return torch.from_numpy(dist_matrix).float()

def get_prot_sm_mask(atom_mask, seq):
    """
    识别出在一个包含蛋白质和小分子的混合序列中，哪些位置是有效的（即具有足够结构信息的），并生成一个统一的有效性掩码
    Parameters
    ----------
    atom_mask : (..., L, Natoms)
    seq : (L)

    Returns
    -------
    mask : (..., L)
    """
    sm_mask = is_atom(seq).to(atom_mask.device) # (L)。用于区分是小分子还是蛋白质
    has_backbone = atom_mask[...,:3].all(dim=-1)  # 蛋白质主链是否完整的掩码
    # has_backbone_prot = has_backbone[...,~sm_mask]
    # n_protein_with_backbone = has_backbone.sum()
    # n_protein = (~sm_mask).sum()
    #assert_that((n_protein/n_protein_with_backbone).item()).is_greater_than(0.8)
    # 只有当一个位置不是小分子，且具有完整主链时，才被判定为有效的蛋白质残基
    mask_prot = has_backbone & ~sm_mask
    # 在小分子的坐标表示中，索引1通常被预留作该分子的“锚定点”（类似于蛋白质的CA原子）
    mask_ca_sm = atom_mask[..., 1] & sm_mask

    mask = mask_prot | mask_ca_sm
    return mask

def xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames):
    """
    Parameters:
        xyz_t (1, T, L, natoms, 3)
        seq_unmasked (B, L)
        atom_frames (1, A, 3, 2)
    Returns:
	    xyz_t_frame (B, T, L, natoms, 3)
    """
    is_sm = is_atom(seq_unmasked[0])  # 判断模板中哪部分是小分子配体
    return xyz_t_to_frame_xyz_sm_mask(xyz_t, is_sm, atom_frames)

def xyz_t_to_frame_xyz_sm_mask(xyz_t, is_sm, atom_frames):
    """
    Parameters:
        xyz_t (1, T, L, natoms, 3)
        is_sm (L)
        atom_frames (1, A, 3, 2)
    Returns:
	xyz_t_frame (B, T, L, natoms, 3)
    """
    # ic(xyz_t.shape, is_sm.shape, atom_frames.shape)
    # xyz_t.shape: torch.Size([1, 1, 194, 36, 3])
    # is_sm.shape: torch.Size([194])
    # atom_frames.shape: torch.Size([1, 29, 3, 2])
    xyz_t_frame = xyz_t.clone()
    atoms = is_sm
    if torch.all(~atoms):
        return xyz_t_frame
    atom_crds_t = xyz_t_frame[:, :, atoms]

    B, T, atom_L, natoms, _ = atom_crds_t.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_t_frame[:, :, atoms, :3] = atom_crds_t.reshape(T, atom_L*natoms, 3)[:, frames_reindex.squeeze(0)]
    return xyz_t_frame


def xyz_frame_from_rotation_mask(xyz, rotation_mask, atom_frames):
    """
    function to get xyz_frame for l1 feature in Structure module
    xyz (1, L, natoms, 3)
    rotation_mask (1, L)
    atom_frames (1, L, 3, 2)
    """
    xyz_frame = xyz.clone()
    if torch.all(~rotation_mask):
        return xyz_frame

    atom_crds = xyz_frame[rotation_mask]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])

    for i in range(atom_L):
        frames_reindex[:, i, :] = (i + atom_frames[..., i, :, 0]) * natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_frame[rotation_mask, :, :3] = atom_crds.reshape(atom_L * natoms, 3)[frames_reindex]
    return xyz_frame

# 基于3个点构建框架
#fd  -  更复杂的版本将角度偏差分配给CA-N和CA-C（从而获得更精确的CB位置）
#fd  -  对输入维度不作任何假设（仅要求最后一个参数为xyz）
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    dims = N.shape[:-1]

    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    u2 = v2 - (torch.einsum('...li, ...li -> ...l', e1, v2)[..., None] * e1)
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[..., None], e2[..., None], e3[..., None]], axis=-1)  # [B,L,3,3] - rotation matrix

    if non_ideal:
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
        cosref = torch.clamp(torch.sum(e1 * v2, dim=-1), min=-1.0, max=1.0)  # cosine of current N-CA-C bond angle
        costgt = chemical.cos_ideal_NCAC.item()
        cos2del = torch.clamp(cosref * costgt + torch.sqrt((1 - cosref * cosref) * (1 - costgt * costgt) + eps), min=-1.0, max=1.0)
        cosdel = torch.sqrt(0.5 * (1 + cos2del) + eps)
        sindel = torch.sign(costgt - cosref) * torch.sqrt(1 - 0.5 * (1 + cos2del) + eps)
        Rp = torch.eye(3, device=N.device).repeat(*dims, 1, 1)
        Rp[..., 0, 0] = cosdel
        Rp[..., 0, 1] = -sindel
        Rp[..., 1, 0] = sindel
        Rp[..., 1, 1] = cosdel

        R = torch.einsum('...ij,...jk->...ik', R, Rp)

    return R, Ca

# 用于修正蛋白质中可能存在几何畸变的原子坐标
def idealize_reference_frame(xyz_in):
    xyz = xyz_in.clone()

    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], non_ideal=True)

    # 不再使用输入中原本的N和C坐标，而是使用了理想坐标加上得到的SE(3)来获得理想化的主链坐标
    xyz[..., 0, :] = torch.einsum('...ij,j->...i', Rs, chemical.init_N.to(device=xyz_in.device) ) + Ts
    xyz[..., 2, :] = torch.einsum('...ij,j->...i', Rs, chemical.init_C.to(device=xyz_in.device) ) + Ts

    return xyz

def same_chain_from_bond_feats(bond_feats):
    """
    返回一个二进制矩阵，根据残基对的键特征，指示它们是否位于同一链上。
    """
    assert(len(bond_feats.shape)==2) # assume no batch dimension
    L = bond_feats.shape[0]
    same_chain = torch.zeros((L,L))
    G = nx.from_numpy_array(bond_feats.detach().cpu().numpy())
    for idx in nx.connected_components(G):
        idx = list(idx)
        for i in idx:
            same_chain[i, idx] = 1
    return same_chain

def TemplFeaturizeFixbb(seq, num_classes=21, conf_1d=None):
    """
    用于固定主链示例的1D特征化模板:
    Parameters:
        seq (torch.tensor, required): Integer sequence
        conf_1d (torch.tensor, optional): 预计算的其余特征
    """
    # L = seq.shape[-1]
    t1d = torch.nn.functional.one_hot(seq, num_classes=num_classes)  # one hot sequence
    if conf_1d is None:
        conf = torch.concat([torch.ones((seq.shape[0], 2)), torch.zeros((seq.shape[0], 2))], dim=-1)  # 序列置信度、结构置信度、热点区域、时间步
    else:
        conf = conf_1d[:, None]  # 升维
    # 将置信度合并到编码后面
    t1d = torch.cat((t1d, conf), dim=-1)
    return t1d

def get_protein_bond_feats(protein_L):
    """ 生成蛋白质残基连接图 """
    bond_feats = torch.zeros((protein_L, protein_L))
    residues = torch.arange(protein_L-1)
    # 构造所有相邻残基对（除对角线外）的存在主链键（5为残基-残基键）
    bond_feats[residues, residues+1] = 5
    bond_feats[residues+1, residues] = 5
    return bond_feats

def delete_leaving_atoms_single_chain(filename, is_leaving, is_str=False):
    """
    根据is_leaving去除指定的原子
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat('sdf')
    obmol = openbabel.OBMol()

    # 将元素名称的第二个字母转换为小写
    if is_str:
        molstring = clean_sdffile(filename.split('\n'), is_str=True)
    else:
        molstring = clean_sdffile(filename)

    obConversion.ReadString(obmol, molstring)

    assert len(is_leaving) == obmol.NumAtoms()
    leaving_indices = torch.tensor(is_leaving).nonzero()
    for i, idx in enumerate(leaving_indices):
        obmol.DeleteAtom(obmol.GetAtom(idx.item() + 1 - i))

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "sdf")
    sdf_string = obConversion.WriteString(obmol)
    return sdf_string

def parse_mol(ligand_path, filetype='sdf', is_str=False, remove_H=True, generate_conformer=True):
    """
    返回sdf文件的openbabel对象、手性信息以及化学键信息
    """
    if type(ligand_path) is str:
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat(filetype)  # 读取的类型，可选mol2、sdf、smi
        obmol = openbabel.OBMol()

        # 将元素名称的第二个字母转换为小写
        if is_str:
            molstring = clean_sdffile(ligand_path.split('\n'), is_str=True)
        else:
            molstring = clean_sdffile(ligand_path.replace('.mol2', '.sdf'))

        obConversion.ReadString(obmol, molstring)
    elif type(ligand_path) is openbabel.OBMol:
        obmol = ligand_path
    else:
        raise ValueError("请输入sdf文件路径或者sdf字符串内容或者OBMol对象")

    """优化3D构象"""
    if generate_conformer:
        builder = openbabel.OBBuilder()
        builder.Build(obmol)
        ff = openbabel.OBForceField.FindForceField("mmff94")  # mmff94立场对于一些元素会失败
        did_setup = ff.Setup(obmol)
        if did_setup:
            ff.FastRotorSearch()
            ff.GetCoordinates(obmol)
        else:
            ff = openbabel.OBForceField.FindForceField("uff")  # 回退到uff立场
            did_setup = ff.Setup(obmol)
            ff.FastRotorSearch()
            ff.GetCoordinates(obmol)

    if remove_H:
        obmol.DeleteHydrogens()
        # 上述方法有时无法捕获所有氢原子
        i = 1
        while i < obmol.NumAtoms() + 1:
            if obmol.GetAtom(i).GetAtomicNum() == 1:
                obmol.DeleteAtom(obmol.GetAtom(i))
            else:
                i += 1

    atom_coords = torch.tensor([[obmol.GetAtom(i).x(), obmol.GetAtom(i).y(), obmol.GetAtom(i).z()]
                                for i in range(1, obmol.NumAtoms() + 1)])  # [atom_nums, 3]
    atomtypes = [chemical.atomnum2atomtype.get(obmol.GetAtom(i).GetAtomicNum())
                 for i in range(1, obmol.NumAtoms() + 1)]  # GetAtomicNum()获取的是原子序数
    atom_tokens = torch.tensor([chemical.aa2num[x] for x in atomtypes])

    if torch.isinf(atom_coords).any().item():
        return None

    # 获取小分子手性信息。[num_chiral,5]。前四个为形成手性约束的四个原子的索引（第一个为中心原子），第五个为目标理想伪二面角。需要注意openbabel的第一个原子的序号为1，但是其id默认从0开始
    chirals = get_chirals(obmol, atom_coords)

    obmol.PerceiveBondOrders()  # 建立键化学信息
    bond_feats, real_bond_feats = get_bond_feats(obmol)  # 生成小分子键特征

    # 创建原子框架
    G = get_nxgraph(obmol)
    atom_frames = get_atom_frames(atom_tokens, G)  # 获取原子索引框架，用于计算全原子fape损失

    return {'obmol': obmol, 'atom_tokens': atom_tokens, 'atom_coords': atom_coords, 'chirals': chirals, 'bond_feats': bond_feats,
            'real_bond_feats': real_bond_feats, 'atom_frames': atom_frames}

def make_openbabel_bond(mol, i, j, order):
    """
    手动创建OpenBabel的键对象。Open Babel 3.x 不推荐使用OBBond添加化学键
    """
    obb = openbabel.OBBond()
    # 指定该键的起点原子和终点原子
    obb.SetBegin(mol.GetAtom(i+1))
    obb.SetEnd(mol.GetAtom(j+1))
    if order == 4:
        obb.SetBondOrder(2)
        obb.SetAromatic()  # 将化学键打上“芳香性”的标记
        # 同时把两端原子也设为芳香更一致
        obb.GetBeginAtom().SetAromatic(True)
        obb.GetEndAtom().SetAromatic(True)
    else:
        obb.SetBondOrder(order)
    return obb

def get_combined_atoms_bonds(combined_molecule):
    L_total = sum([bf.shape[0] for bf in combined_molecule['real_bond_feats_list']])
    real_bond_feats = torch.zeros((L_total, L_total)).long()
    bond_feats = torch.zeros((L_total, L_total)).long()
    offset = 0
    for bf in combined_molecule['real_bond_feats_list']:
        L = bf.shape[0]
        real_bond_feats[offset:offset + L, offset:offset + L] = bf
        bond_feats[offset:offset + L, offset:offset + L] = bf
        offset += L
    bond_feats[offset - L:offset, offset - L:offset] = combined_molecule['bond_feats']
    xyz = torch.concat(combined_molecule['xyzs'], dim=0)
    xyz_mask = torch.concat(combined_molecule['cova_atom_mask'], dim=0)
    return real_bond_feats, bond_feats, xyz, xyz_mask

def make_obmol_from_atoms_bonds(msa, xyz, bond_feats, add_H=False, extra_bonds=None, return_sdf_str=False):
    # 创建一个可编辑的分子对象
    mol = Chem.RWMol()

    # 添加原子
    for element in msa:
        atom_num = chemical.atomtype2atomnum[element]
        atom = Chem.Atom(atom_num)
        mol.AddAtom(atom)

    # 设置坐标
    conf = Chem.Conformer(len(msa))
    for i in range(len(msa)):
        conf.SetAtomPosition(i, (float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
    mol.AddConformer(conf)

    # 添加化学键
    first_index, second_index = bond_feats.nonzero(as_tuple=True)
    added_bonds = set()  # 防止重复添加双向键
    for i, j in zip(first_index, second_index):
        idx1, idx2 = i.item(), j.item()
        if idx1 < idx2:  # 保证唯一性
            order = int(bond_feats[i, j].item())
            bt = rdkit_bond_type_lookup.get(order, Chem.rdchem.BondType.SINGLE)
            mol.AddBond(idx1, idx2, bt)
            added_bonds.add(tuple(sorted((idx1, idx2))))

    # 添加共价键
    if extra_bonds is not None:
        for idx_i, idx_j in extra_bonds:
            if tuple(sorted((int(idx_i), int(idx_j)))) not in added_bonds:
                mol.AddBond(int(idx_i), int(idx_j), Chem.rdchem.BondType.SINGLE)  # 共价键均为单键

                # 共价结合过程中，通常会发生化学键的重组，因此需要检查共价连接的原子价态
                for check_idx in [idx_i, idx_j]:
                    atom = mol.GetAtomWithIdx(check_idx)
                    # 必须更新缓存
                    mol.UpdatePropertyCache(strict=False)

                    # 获取该原子在周期表中的标准价态
                    # 也可以简单写死：max_val = 4 if atom.GetSymbol() == 'C' else ...
                    periodic_table = Chem.GetPeriodicTable()
                    max_val = periodic_table.GetDefaultValence(atom.GetAtomicNum())

                    # 如果显性价键超过了标准价态
                    while atom.GetExplicitValence() > max_val:
                        found_multi_bond = False
                        # 查找是否有双键或三键可以降级
                        for b in atom.GetBonds():
                            # 跳过刚刚添加的共价键
                            if tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) == tuple(sorted((idx_i, idx_j))):
                                continue

                            if b.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC]:
                                b.SetBondType(Chem.rdchem.BondType.SINGLE)
                                found_multi_bond = True
                                break
                            elif b.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                                b.SetBondType(Chem.rdchem.BondType.DOUBLE)
                                found_multi_bond = True
                                break

                        if not found_multi_bond:
                            break
                        mol.UpdatePropertyCache(strict=False)

    # 自动处理芳香性、化合价检查等
    res_mol = mol.GetMol()

    if add_H:
        # RDKit 增加氢原子通常需要先 Sanitization
        Chem.SanitizeMol(res_mol)
        res_mol = Chem.AddHs(res_mol, addCoords=True)
    else:
        Chem.SanitizeMol(res_mol)

    # 验证芳香性
    # for bond in res_mol.GetBonds():
    #     print(f"Bond {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}: IsAromatic={bond.GetIsAromatic()}")

    smiles2 = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)  # 保留手性/顺反

    return mol if not return_sdf_str else Chem.MolToMolBlock(res_mol), smiles2

def make_obmol_from_atoms_bonds_use_openbabel(msa, xyz, real_bond_feats, bond_feats, add_H=False, extra_bonds=None, return_sdf=False):
    """
    根据元素类型、坐标和化学键类型创建新的openbabel对象
    """
    mol = openbabel.OBMol()
    mol.BeginModify()

    for i, element in enumerate(msa):
        atomnum = chemical.atomtype2atomnum[element]  # 原子序数
        a = mol.NewAtom()
        a.SetAtomicNum(atomnum)
        a.SetVector(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))

    first_index, second_index = real_bond_feats.nonzero(as_tuple=True)
    # 添加化学键
    for i, j in zip(first_index, second_index):
        if i < j:
            order = real_bond_feats[i, j].item()
            mol.AddBond(i.item() + 1, j.item() + 1, order)

    if extra_bonds is not None:
        order = 1  # 所有共价键均为单键
        for bond in extra_bonds:
            mol.AddBond(bond[0] + 1, bond[1] + 1, order)

    # mol.PerceiveBondOrders()  # 重新感知化学键（包括芳香性）

    # 显式设置芳香键
    for i, j in zip(first_index, second_index):
        order = bond_feats[i, j].item()
        if order == 4:
            b = mol.GetBond(int(i) + 1, int(j) + 1)
            b.SetAromatic(True)
            # 同时把两端原子也设为芳香
            b.GetBeginAtom().SetAromatic(True)
            b.GetEndAtom().SetAromatic(True)

    mol.SetAromaticPerceived(True)

    p_mol = pybel.Molecule(mol)
    smiles = p_mol.write("smi").strip()
    if add_H:
        p_mol.removeh()
        p_mol.addh()  # 添加氢原子

    mol.EndModify()  # 通知babel结束编辑

    # for b in openbabel.OBMolBondIter(mol):
    #     print(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.IsAromatic(), b.GetBondOrder())

    # 获取小分子手性信息。[num_chiral,5]。前四个为形成手性约束的四个原子的索引（第一个为中心原子），第五个为目标理想伪二面角。需要注意openbabel的第一个原子的序号为1，但是其id默认从0开始
    chirals = get_chirals(mol, xyz)

    bond_feats, _ = get_bond_feats(mol)  # 生成小分子键特征

    # 创建原子框架
    atomtypes = [chemical.atomnum2atomtype.get(mol.GetAtom(i).GetAtomicNum())
                 for i in range(1, mol.NumAtoms() + 1)]  # GetAtomicNum()获取的是原子序数
    G = get_nxgraph(mol)
    atom_tokens = torch.tensor([chemical.aa2num[x] for x in atomtypes])
    atom_frames = get_atom_frames(atom_tokens, G)  # 获取原子索引框架，用于计算全原子fape损失

    if return_sdf:
        conv = openbabel.OBConversion()
        conv.SetOutFormat("sdf")
        sdf_str = conv.WriteString(mol)
        return sdf_str, mol, chirals, bond_feats, atom_frames, smiles

    return mol, chirals, bond_feats, atom_frames, smiles

def align_molecule_indices(template_mol, target_mol, add_H=False, remove_H=False):
    """
    将 target_mol 的原子顺序调整为与 template_mol 一致
    """
    # 确保两个分子都去氢或都加氢（保证重原子匹配一致）
    if remove_H:
        template_mol = Chem.RemoveAllHs(template_mol)
        target_mol = Chem.RemoveAllHs(target_mol)
    if add_H:
        template_mol = Chem.AddHs(template_mol)
        target_mol = Chem.AddHs(target_mol)

    """由于只是对齐原子顺序，因此将所有形式电荷清除掉是合理的"""
    for atom in template_mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)

    # 基于模板修复。由于PDBBind+提供的sdf文件可能会存在化学键连接错误，因此使用理想sdf进行修复
    fixed_mol = AllChem.AssignBondOrdersFromTemplate(target_mol, template_mol)

    # Chem.SanitizeMol(fixed_mol)
    # Chem.SanitizeMol(target_mol)

    # 在 target 中寻找 template 的匹配模式。这会返回一个元组，其顺序对应 template 的原子索引
    match_indices = target_mol.GetSubstructMatch(fixed_mol, useChirality=True)

    if not match_indices:
        # 如果匹配不成功，尝试忽略键级（有时 PDB 的键级定义有误）
        params = Chem.SubstructMatchParameters()
        params.useChirality = False  # 忽略手性差异
        params.useBondOrder = False  # 忽略键级差异
        match_indices = target_mol.GetSubstructMatch(fixed_mol, params)

    if len(match_indices) != fixed_mol.GetNumAtoms():
        return fixed_mol, None

    # 根据匹配到的索引顺序重新排列 target_mol 的原子
    new_order = list(match_indices)
    reordered_mol = Chem.RenumberAtoms(target_mol, new_order)

    return fixed_mol, reordered_mol

def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def lddt_unbin(pred_lddt):
    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)


def get_frames(xyz_mask, seq, frame_indices, atom_frames):
    L = xyz_mask.shape[0]
    ligand_L = atom_frames.shape[0]
    frames = frame_indices[seq]
    atoms = seq > chemical.NNAPROTAAS
    if torch.any(atoms):
        frames[atoms.nonzero().flatten(), 0] = atom_frames

    """一个局部坐标系有效，必须要求它依赖的3个原子都被解析出来"""
    frame_mask = ~torch.all(frames[...,0, :] == frames[...,1, :], axis=-1)
    # 根据xyz_mask进一步排除掉无效的框架
    res_offset = frames[..., 0]  # (L, 6, 3)
    atom_idx = frames[..., 1]  # (L, 6, 3)
    # 构建当前残基的绝对索引
    base_res_idx = torch.arange(L, device=xyz_mask.device).view(L, 1, 1)
    target_res_idx = base_res_idx + res_offset
    target_res_idx = target_res_idx.clamp(min=0, max=L - 1)  # 防止索引越界
    atom_exists_mask = xyz_mask[target_res_idx, atom_idx]  # 根据创建的原子索引去确认原子是否存在
    # valid_res_mask = (target_res_idx >= 0) & (target_res_idx < L)  # 防止未来引入跨残基frame时发生索引越界
    # atom_exists_mask = atom_exists_mask & valid_res_mask
    frame_mask &= torch.all(atom_exists_mask, dim=-1)

    return frames, frame_mask


def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None, :, None, None] * pred_pae, dim=1)
