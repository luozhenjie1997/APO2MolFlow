import torch
import torch.nn as nn
import numpy as np
import copy
import dgl
import model.utils.chemical as chemical
from model.utils.utils import rigid_from_3_points, idealize_reference_frame

def init_lecun_normal(module, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

def init_lecun_normal_param(weight, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
    return weight

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 为不同的网络层设置不同的权重衰减
def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        """
        偏置项通常不应用权重衰减，因为它们的作用是调整激活值的均值，而不是直接参与特征提取。
        BatchNorm、LayerNorm等，这些层的参数（如缩放因子和偏移量）也不应该应用权重衰减。因为这些参数的作用是调整激活值的分布，而不是直接参与特征提取。
        对这些参数应用权重衰减可能会破坏归一化的效果，影响模型的训练和收敛。
        """
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x


def get_res_atom_dist(idx, bond_feats, dist_matrix, sm_mask, is_atomize_protein, minpos_res=-32, maxpos_res=32, maxpos_atom=8):
    '''
    Calculates residue and atom bond distances of protein/SM complex. Used for positional
    embedding and structure module. 2nd version (2022-9-19); handles atomized proteins.

    Input:
        - idx: residue index (B, L)
        - bond_feats: bond features (B, L, L)
        - dist_matrix: precomputed bond distances (B, L, L) NOTE: need to run nan_to_num to remove infinities
        - sm_mask: boolean feature (L). True if a position represents atom, False otherwise
        - minpos_res: minimum value of residue distances
        - maxpos_res: maximum value of residue distances
        - maxpos_atom: maximum value of atom bond distances

    Output:
        - res_dist: residue distance (B, L, L)
        - atom_dist: atom bond distance (B, L, L)
    '''
    bond_feats = bond_feats[0]  # assume batch = 1
    L = bond_feats.shape[0]
    device = bond_feats.device

    sm_mask_2d = sm_mask[None, :] * sm_mask[:, None]
    prot_mask_2d = (~sm_mask[None, :]) * (~sm_mask[:, None])
    inter_mask_2d = (~sm_mask[None, :]) * (sm_mask[:, None]) + (sm_mask[None, :]) * (~sm_mask[:, None])

    # protein residue distances
    res_dist_prot = torch.clamp(idx[0, None, :] - idx[0, :, None],
                                min=minpos_res, max=maxpos_res)  # (L, L) intra-protein
    res_dist_sm = torch.full((L, L), maxpos_res + 1, device=device)  # (L, L) with "unknown" res. dist. token

    # small molecule atom bond graph
    atom_dist_sm = torch.nan_to_num(dist_matrix, posinf=maxpos_atom)[0].long()  # this comes through the dataloader so it is batched
    atom_dist_prot = torch.full((L, L), maxpos_atom + 1, device=device)

    res_dist_inter = torch.full((L, L), maxpos_res, device=device)
    atom_dist_inter = torch.full((L, L), maxpos_atom, device=device)

    # 处理共价结合的情况（fd new impl）
    i_s, j_s = torch.where(bond_feats == 6)
    if i_s.shape[0] > 0 and j_s.shape[0] > 0:
        sm_mask = sm_mask | is_atomize_protein
        i_sm = i_s[sm_mask[i_s]]
        i_prot = j_s[sm_mask[i_s]]
        closest_prot_res = i_prot[torch.argmin(atom_dist_sm[sm_mask, :][:, i_sm], dim=-1)]
        res_dist_inter[sm_mask, :] = res_dist_prot[closest_prot_res, :]
        res_dist_inter[:, sm_mask] = res_dist_prot[:, closest_prot_res]

        closest_atom = i_sm[torch.argmin(torch.abs(res_dist_prot[~sm_mask, :][:, i_prot]), dim=-1)]
        atom_dist_inter[~sm_mask, :] = atom_dist_sm[closest_atom, :] + 1
        atom_dist_inter[:, ~sm_mask] = atom_dist_sm[:, closest_atom] + 1

    res_dist = res_dist_prot * prot_mask_2d + res_dist_inter * inter_mask_2d + res_dist_sm * sm_mask_2d
    atom_dist = atom_dist_prot * prot_mask_2d + atom_dist_inter * inter_mask_2d + atom_dist_sm * sm_mask_2d

    return res_dist[None], atom_dist[None]  # add batch dim.


def get_seqsep_protein_sm(idx, bond_feats, dist_matrix, sm_mask, is_atomize_protein):
    '''
    Sequence separation features for protein-SM complex

    Input:
        - idx: residue indices of given sequence (B,L)
        - bond_feats: bond features (B, L, L)
        - dist_matrix: precomputed bond distances (B, L, L) NOTE: need to run nan_to_num to remove infinities
        - sm_mask: boolean feature True if a position represents atom, False if residue (B, L)

    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
            -1 or 1 for bonded protein residues
            1 for bonded SM atoms or residue-atom bonds
            0 elsewhere
    '''
    sm_mask = sm_mask[0]  # assume batch = 1
    res_dist, atom_dist = get_res_atom_dist(idx, bond_feats, dist_matrix, sm_mask, is_atomize_protein)

    sm_mask_2d = sm_mask[None, :] * sm_mask[:, None]
    prot_mask_2d = (~sm_mask[None, :]) * (~sm_mask[:, None])
    inter_mask_2d = (~sm_mask[None, :]) * (sm_mask[:, None]) + (sm_mask[None, :]) * (~sm_mask[:, None])

    res_dist[(res_dist > 1) | (res_dist < -1)] = 0.0
    atom_dist[(atom_dist > 1)] = 0.0

    seqsep = sm_mask_2d * atom_dist + prot_mask_2d * res_dist + inter_mask_2d * (bond_feats == 6)

    return seqsep.unsqueeze(-1)

def rbf(D, D_min=0.0, D_count=64, D_sigma=0.5):
    # Distance radial basis function
    D_max = D_min + (D_count-1) * D_sigma
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def make_topk_graph(xyz, pair, idx, top_k=128, nlocal=33, topk_incl_local=True, eps=1e-6):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device

    # distance map from current CA coordinates
    D = torch.cdist(xyz, xyz) + torch.eye(L, device=device).unsqueeze(0)*9999.9  # (B, L, L)

    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*9999.9

    if (topk_incl_local):
        D = D + sep*eps
        D[sep<nlocal] = 0.0

        # get top_k neighbors
        D_neigh, E_idx = torch.topk(D, min(top_k, L-1), largest=False) # shape of E_idx: (B, L, top_k)
        topk_matrix = torch.zeros((B, L, L), device=device)
        topk_matrix.scatter_(2, E_idx, 1.0)
        cond = topk_matrix > 0.0

    else:

        D = D + sep*eps

        # get top_k neighbors
        D_neigh, E_idx = torch.topk(D, min(top_k, L-1), largest=False) # shape of E_idx: (B, L, top_k)
        topk_matrix = torch.zeros((B, L, L), device=device)
        topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < nlocal)
    b,i,j = torch.where(cond)

    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['rel_pos'] = (xyz[b,j,:] - xyz[b,i,:]).detach() # no gradient through basis function

    return G, pair[b,i,j][...,None]


def make_full_graph(xyz, pair, idx):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device

    # seq sep
    sep = idx[:, None, :] - idx[:, :, None]
    b, i, j = torch.where(sep.abs() > 0)
    src = b * L + i
    tgt = b * L + j
    G = dgl.graph((src, tgt), num_nodes=B * L).to(device)
    G.edata['rel_pos'] = (xyz[b, j, :] - xyz[b, i, :])  # .detach() # no gradient through basis function
    return G, pair[b, i, j][..., None]

# 绕x轴旋转
def make_rotX(angs, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)

    RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
    RTs[:, :, 1, 2] = -angs[:, :, 1] / NORM
    RTs[:, :, 2, 1] = angs[:, :, 1] / NORM
    RTs[:, :, 2, 2] = angs[:, :, 0] / NORM
    return RTs

# 绕z轴旋转
def make_rotZ(angs, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)

    RTs[:, :, 0, 0] = angs[:,:,0] / NORM
    RTs[:, :, 0, 1] = -angs[:,:,1] / NORM
    RTs[:, :, 1, 0] = angs[:,:,1] / NORM
    RTs[:, :, 1, 1] = angs[:,:,0] / NORM
    return RTs

# 绕任意轴旋转
def make_rot_axis(angs, u, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)

    ct = angs[:, :, 0] / NORM
    st = angs[:, :, 1] / NORM
    u0 = u[:, :, 0]
    u1 = u[:, :, 1]
    u2 = u[:, :, 2]

    RTs[:, :, 0, 0] = ct + u0 * u0 * (1 - ct)
    RTs[:, :, 0, 1] = u0 * u1 * (1 - ct) - u2 * st
    RTs[:, :, 0, 2] = u0 * u2 * (1 - ct) + u1 * st
    RTs[:, :, 1, 0] = u0 * u1 * (1 - ct) + u2 * st
    RTs[:, :, 1, 1] = ct + u1 * u1 * (1 - ct)
    RTs[:, :, 1, 2] = u1 * u2 * (1 - ct) - u0 * st
    RTs[:, :, 2, 0] = u0 * u2 * (1 - ct) - u1 * st
    RTs[:, :, 2, 1] = u1 * u2 * (1 - ct) + u0 * st
    RTs[:, :, 2, 2] = ct + u2 * u2 * (1 - ct)
    return RTs

class XYZConverter(nn.Module):
    def __init__(self):
        super(XYZConverter, self).__init__()

        self.register_buffer("torsion_indices", chemical.torsion_indices, persistent=False)
        self.register_buffer("torsion_can_flip", chemical.torsion_can_flip.to(torch.int32), persistent=False)
        self.register_buffer("ref_angles", chemical.reference_angles, persistent=False)
        self.register_buffer("base_indices", chemical.base_indices, persistent=False)
        self.register_buffer("RTs_in_base_frame", chemical.RTs_by_torsion, persistent=False)
        self.register_buffer("xyzs_in_base_frame", chemical.xyzs_in_base_frame, persistent=False)

    def compute_all_atom(self, seq, xyz, alphas, non_ideal=False):
        B, L = xyz.shape[:2]

        # 蛋白质主链局部坐标系
        Rs, Ts = rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :], non_ideal=non_ideal)

        RTF0 = torch.eye(4).repeat(B, L, 1, 1).to(device=Rs.device)

        # bb
        RTF0[:, :, :3, :3] = Rs
        RTF0[:, :, :3, 3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 0, :], make_rotX(alphas[:, :, 0, :]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 1, :], make_rotX(alphas[:, :, 1, :]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 2, :], make_rotX(alphas[:, :, 2, :]))

        # CB原子位置修正
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5 * (basexyzs[:, :, 2, :3] + basexyzs[:, :, 0, :3])
        CAr = (basexyzs[:, :, 1, :3])
        CBr = (basexyzs[:, :, 4, :3])
        CBrotaxis1 = (CBr - CAr).cross(NCr - CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True) + 1e-8

        # CB twist
        NCp = basexyzs[:, :, 2, :3] - basexyzs[:, :, 0, :3]
        NCpp = NCp - torch.sum(NCp * NCr, dim=-1, keepdim=True) / torch.sum(NCr * NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr - CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True) + 1e-8

        CBrot1 = make_rot_axis(alphas[:, :, 7, :], CBrotaxis1)
        CBrot2 = make_rot_axis(alphas[:, :, 8, :], CBrotaxis2)

        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, CBrot1, CBrot2)

        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim',
            RTF8,
            self.RTs_in_base_frame[seq, 3, :],
            make_rotX(alphas[:, :, 3, :]),
            make_rotZ(alphas[:, :, 9, :]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF4, self.RTs_in_base_frame[seq, 4, :], make_rotX(alphas[:, :, 4, :]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF5, self.RTs_in_base_frame[seq, 5, :], make_rotX(alphas[:, :, 5, :]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF6, self.RTs_in_base_frame[seq, 6, :], make_rotX(alphas[:, :, 6, :]))

        # ignore RTs_in_base_frame[seq,7:9,:] and alphas[:,:,10:12,:]

        # 堆叠扭转角后计算全原子坐标
        RTframes = torch.stack((RTF0, RTF1, RTF2, RTF3, RTF4, RTF5, RTF6, RTF7, RTF8), dim=2)
        xyzs = torch.einsum(
            'brtij,brtj->brti',
            RTframes.gather(2, self.base_indices[seq][..., None, None].repeat(1, 1, 1, 4, 4)), basexyzs
        )

        return RTframes, xyzs[..., :3]

    def get_tor_mask(self, seq, mask_in=None):
        B, L = seq.shape[:2]

        tors_mask = self.torsion_indices[seq, :, -1] > 0

        if mask_in != None:  # 一般在训练阶段才会传入，表示哪些原子是真实存在的
            ts = self.torsion_indices[seq]
            bs = torch.arange(B, device=seq.device)[:, None, None, None]
            rs = torch.arange(L, device=seq.device)[None, :, None, None] - (ts < 0) * 1  # ts<-1 ==> prev res
            ts = torch.abs(ts)
            tors_mask *= mask_in[bs, rs, ts].all(dim=-1)

        return tors_mask

    def get_torsions(self, xyz_in, seq, mask_in=None):
        B, L = xyz_in.shape[:2]

        tors_mask = self.get_tor_mask(seq, mask_in)
        # 在计算扭转角之前，先对给定坐标进行理想化处理
        xyz = idealize_reference_frame(seq, xyz_in)

        ts = self.torsion_indices[seq]
        bs = torch.arange(B, device=xyz_in.device)[:, None, None, None]
        xs = torch.arange(L, device=xyz_in.device)[None, :, None, None] - (ts < 0) * 1  # ts<-1 ==> prev res
        ys = torch.abs(ts)
        xyzs_bytor = xyz[bs, xs, ys, :]

        # 初始化输出张量
        torsions = torch.zeros((B, L, chemical.NTOTALDOFS, 2), device=xyz_in.device)

        # 提取侧链chi1-chi4和主链角度
        torsions[..., :7, :] = chemical.th_dih(
            xyzs_bytor[..., :7, 0, :], xyzs_bytor[..., :7, 1, :], xyzs_bytor[..., :7, 2, :], xyzs_bytor[..., :7, 3, :]
        )
        torsions[:, :, 2, :] = -1 * torsions[:, :, 2, :]  # shift psi by pi

        # protein angles
        # CB bend
        NC = 0.5 * (xyz[:, :, 0, :3] + xyz[:, :, 2, :3])
        CA = xyz[:, :, 1, :3]
        CB = xyz[:, :, 4, :3]
        t = chemical.th_ang_v(CB - CA, NC - CA)
        t0 = self.ref_angles[seq][..., 0, :]
        torsions[:, :, 7, :] = torch.stack(
            (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
            dim=-1)

        # CB twist
        NCCA = NC - CA
        NCp = xyz[:, :, 2, :3] - xyz[:, :, 0, :3]
        NCpp = NCp - torch.sum(NCp * NCCA, dim=-1, keepdim=True) / torch.sum(NCCA * NCCA, dim=-1, keepdim=True) * NCCA
        t = chemical.th_ang_v(CB - CA, NCpp)
        t0 = self.ref_angles[seq][..., 1, :]
        torsions[:, :, 8, :] = torch.stack(
            (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
            dim=-1)

        # CG bend
        CG = xyz[:, :, 5, :3]
        t = chemical.th_ang_v(CG - CB, CA - CB)
        t0 = self.ref_angles[seq][..., 2, :]
        torsions[:, :, 9, :] = torch.stack(
            (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
            dim=-1)

        mask0 = (torch.isnan(torsions[..., 0])).nonzero()
        mask1 = (torch.isnan(torsions[..., 1])).nonzero()
        torsions[mask0[:, 0], mask0[:, 1], mask0[:, 2], 0] = 1.0
        torsions[mask1[:, 0], mask1[:, 1], mask1[:, 2], 1] = 0.0

        # alt chis
        torsions_alt = torsions.clone()
        torsions_alt[self.torsion_can_flip[seq, :].to(torch.bool)] *= -1

        # torsions to restrain to 0 or 180 degree
        # (this should be specified in chemical?)
        tors_planar = torch.zeros((B, L, chemical.NTOTALDOFS), dtype=torch.bool, device=xyz_in.device)
        tors_planar[:, :, 5] = seq == chemical.aa2num['TYR']  # TYR chi 3 should be planar

        return torsions, torsions_alt, tors_mask, tors_planar

"""用于在Dataloader中使用"""
class XYZConverter_loader():
    def __init__(self):
        super(XYZConverter_loader, self).__init__()

        self.torsion_indices = chemical.torsion_indices
        self.torsion_can_flip = chemical.torsion_can_flip.to(torch.int32)
        self.ref_angles = chemical.reference_angles
        self.base_indices = chemical.base_indices
        self.RTs_in_base_frame = chemical.RTs_by_torsion
        self.xyzs_in_base_frame = chemical.xyzs_in_base_frame

    def compute_all_atom(self, seq, xyz, alphas):
        B, L = xyz.shape[:2]

        # 蛋白质主链局部坐标系
        Rs, Ts = rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :])

        RTF0 = torch.eye(4).repeat(B, L, 1, 1).to(device=Rs.device)

        # bb
        RTF0[:, :, :3, :3] = Rs
        RTF0[:, :, :3, 3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 0, :], make_rotX(alphas[:, :, 0, :]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 1, :], make_rotX(alphas[:, :, 1, :]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 2, :], make_rotX(alphas[:, :, 2, :]))

        # CB原子位置修正
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5 * (basexyzs[:, :, 2, :3] + basexyzs[:, :, 0, :3])
        CAr = (basexyzs[:, :, 1, :3])
        CBr = (basexyzs[:, :, 4, :3])
        CBrotaxis1 = (CBr - CAr).cross(NCr - CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True) + 1e-8

        # CB twist
        NCp = basexyzs[:, :, 2, :3] - basexyzs[:, :, 0, :3]
        NCpp = NCp - torch.sum(NCp * NCr, dim=-1, keepdim=True) / torch.sum(NCr * NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr - CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True) + 1e-8

        CBrot1 = make_rot_axis(alphas[:, :, 7, :], CBrotaxis1)
        CBrot2 = make_rot_axis(alphas[:, :, 8, :], CBrotaxis2)

        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, CBrot1, CBrot2)

        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim',
            RTF8,
            self.RTs_in_base_frame[seq, 3, :],
            make_rotX(alphas[:, :, 3, :]),
            make_rotZ(alphas[:, :, 9, :]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF4, self.RTs_in_base_frame[seq, 4, :], make_rotX(alphas[:, :, 4, :]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF5, self.RTs_in_base_frame[seq, 5, :], make_rotX(alphas[:, :, 5, :]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF6, self.RTs_in_base_frame[seq, 6, :], make_rotX(alphas[:, :, 6, :]))

        # ignore RTs_in_base_frame[seq,7:9,:] and alphas[:,:,10:12,:]

        # 堆叠扭转角后计算全原子坐标
        RTframes = torch.stack((RTF0, RTF1, RTF2, RTF3, RTF4, RTF5, RTF6, RTF7, RTF8), dim=2)
        xyzs = torch.einsum(
            'brtij,brtj->brti',
            RTframes.gather(2, self.base_indices[seq][..., None, None].repeat(1, 1, 1, 4, 4)), basexyzs
        )

        return RTframes, xyzs[..., :3]

    def get_tor_mask(self, seq, mask_in=None):
        B, L = seq.shape[:2]

        tors_mask = self.torsion_indices[seq, :, -1] > 0

        if mask_in != None:  # 一般在训练阶段才会传入，表示哪些原子是真实存在的
            ts = self.torsion_indices[seq]
            bs = torch.arange(B, device=seq.device)[:, None, None, None]
            rs = torch.arange(L, device=seq.device)[None, :, None, None] - (ts < 0) * 1  # ts<-1 ==> prev res
            ts = torch.abs(ts)
            tors_mask *= mask_in[bs, rs, ts].all(dim=-1)

        return tors_mask

    def get_torsions(self, xyz_in, seq, mask_in=None):
        B, L = xyz_in.shape[:2]

        tors_mask = self.get_tor_mask(seq, mask_in)
        # 在计算扭转角之前，先对给定坐标进行理想化处理
        xyz = idealize_reference_frame(xyz_in)

        ts = self.torsion_indices[seq]
        bs = torch.arange(B, device=xyz_in.device)[:, None, None, None]
        xs = torch.arange(L, device=xyz_in.device)[None, :, None, None] - (ts < 0) * 1  # ts<-1 ==> prev res
        ys = torch.abs(ts)
        xyzs_bytor = xyz[bs, xs, ys, :]

        # 初始化输出张量
        torsions = torch.zeros((B, L, chemical.NTOTALDOFS, 2), device=xyz_in.device)

        # 提取侧链chi1-chi4和主链角度
        torsions[..., :7, :] = chemical.th_dih(
            xyzs_bytor[..., :7, 0, :], xyzs_bytor[..., :7, 1, :], xyzs_bytor[..., :7, 2, :], xyzs_bytor[..., :7, 3, :]
        )
        torsions[:, :, 2, :] = -1 * torsions[:, :, 2, :]  # shift psi by pi

        # protein angles
        # CB bend
        NC = 0.5 * (xyz[:, :, 0, :3] + xyz[:, :, 2, :3])
        CA = xyz[:, :, 1, :3]
        CB = xyz[:, :, 4, :3]
        t = chemical.th_ang_v(CB - CA, NC - CA)
        t0 = self.ref_angles[seq][..., 0, :]
        torsions[:, :, 7, :] = torch.stack(
            (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
            dim=-1)

        # CB twist
        NCCA = NC - CA
        NCp = xyz[:, :, 2, :3] - xyz[:, :, 0, :3]
        NCpp = NCp - torch.sum(NCp * NCCA, dim=-1, keepdim=True) / torch.sum(NCCA * NCCA, dim=-1, keepdim=True) * NCCA
        t = chemical.th_ang_v(CB - CA, NCpp)
        t0 = self.ref_angles[seq][..., 1, :]
        torsions[:, :, 8, :] = torch.stack(
            (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
            dim=-1)

        # CG bend
        CG = xyz[:, :, 5, :3]
        t = chemical.th_ang_v(CG - CB, CA - CB)
        t0 = self.ref_angles[seq][..., 2, :]
        torsions[:, :, 9, :] = torch.stack(
            (torch.sum(t * t0, dim=-1), t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]),
            dim=-1)

        mask0 = (torch.isnan(torsions[..., 0])).nonzero()
        mask1 = (torch.isnan(torsions[..., 1])).nonzero()
        torsions[mask0[:, 0], mask0[:, 1], mask0[:, 2], 0] = 1.0
        torsions[mask1[:, 0], mask1[:, 1], mask1[:, 2], 1] = 0.0

        # alt chis
        torsions_alt = torsions.clone()
        torsions_alt[self.torsion_can_flip[seq, :].to(torch.bool)] *= -1

        # torsions to restrain to 0 or 180 degree
        # (this should be specified in chemical?)
        tors_planar = torch.zeros((B, L, chemical.NTOTALDOFS), dtype=torch.bool, device=xyz_in.device)
        tors_planar[:, :, 5] = seq == chemical.aa2num['TYR']  # TYR chi 3 should be planar

        return torsions, torsions_alt, tors_mask, tors_planar
