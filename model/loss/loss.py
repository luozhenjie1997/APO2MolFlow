import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.optimize import linear_sum_assignment
from model.utils.chemical import NFRAMES
from model.utils.kinematics import get_dih, get_ang
from model.utils.utils import lddt_unbin, pae_unbin, rigid_from_3_points, find_all_paths_of_length_n, find_all_rigid_groups

#fd more efficient LJ loss
class LJLoss(torch.autograd.Function):
    @staticmethod
    def ljVdV(deltas, sigma, epsilon, lj_lin, eps):
        # deltas - (N,natompair,3)
        N = deltas.shape[0]

        dist = torch.sqrt( torch.sum ( torch.square( deltas ), dim=-1 ) + eps )
        linpart = dist<lj_lin*sigma[None]
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma.repeat(N,1)[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon.repeat(N,1)[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])

        # works for linpart too
        dljEdd_over_r = epsilon * (-12 * sd12/deff + 12 * sd6/deff) / (dist)

        return ljE.sum(dim=-1), dljEdd_over_r

    @staticmethod
    def forward(
        ctx, xs, seq, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds,
        lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75,
        eps=1e-8, training=True
    ):
        N, L, A = xs.shape[:3]
        assert (N==1) # see comment below

        # ds_res = torch.sqrt( torch.sum ( torch.square(
        #     xs.detach()[:,:,None,1,:]-xs.detach()[:,None,:,1,:]), dim=-1 ))
        rs = torch.triu_indices(L,L,0, device=xs.device)
        ri,rj = rs[0],rs[1]

        # batch during inference for huge systems
        BATCHSIZE = 65536//N

        ljval = 0
        dljEdx = torch.zeros_like(xs, dtype=torch.float)

        for i_batch in range((len(ri)-1)//BATCHSIZE + 1):
            idx = torch.arange(
                i_batch*BATCHSIZE,
                min( (i_batch+1)*BATCHSIZE, len(ri)),
                device=xs.device
            )
            rii,rjj = ri[idx],rj[idx] # residue pairs we consider

            ridx,ai,aj = (
                aamask[seq[rii]][:,:,None]*aamask[seq[rjj]][:,None,:]
            ).nonzero(as_tuple=True)

            deltas = xs[:,rii,:,None,:]-xs[:,rjj,None,:,:] # N,BATCHSIZE,Natm,Natm,3
            seqi,seqj = seq[rii[ridx]], seq[rjj[ridx]]

            mask = torch.ones_like(ridx, dtype=torch.bool) # are atoms defined?

            # mask out atom pairs from too-distant residues (C-alpha dist > 24A)
            ca_dist = torch.linalg.norm(deltas[:,:,1,1],dim=-1)
            mask *= (ca_dist[:,ridx]<24).any(dim=0)  # will work for batch>1 but very inefficient

            intrares = (rii[ridx]==rjj[ridx])
            mask[intrares*(ai<aj)] = False  # upper tri (atoms)

            ## count-pair
            # a) intra-protein
            mask[intrares] *= num_bonds[seqi[intrares],ai[intrares],aj[intrares]]>=4
            pepbondres = ri[ridx]+1==rj[ridx]
            mask[pepbondres] *= (
                num_bonds[seqi[pepbondres],ai[pepbondres],2]
                + num_bonds[seqj[pepbondres],0,aj[pepbondres]]
                + 1) >=4

            # b) intra-ligand
            atommask = (ai==1)*(aj==1)
            dist_matrix = torch.nan_to_num(dist_matrix, posinf=4.0) #NOTE: need to run nan_to_num to remove infinities
            resmask = (dist_matrix[0,rii,rjj] >= 4) # * will only work for batch=1
            mask[atommask] *= resmask[ ridx[atommask] ]

            # c) protein/ligand
            ##fd NOTE1: changed 6->5 in masking (atom 5 is CG which should always be 4+ bonds away from connected atom)
            ##fd NOTE2: this does NOT work correctly for nucleic acids
            ##fd     for NAs atoms 0-4 are masked, but also 5,7,8 and 9 should be masked!
            bbatommask = (ai<5)*(aj<5)
            resmask = (bond_feats[0,rii,rjj] != 6) # * will only work for batch=1
            mask[bbatommask] *= resmask[ ridx[bbatommask] ]

            # apply mask.  only interactions to be scored remain
            ai,aj,seqi,seqj,ridx = ai[mask],aj[mask],seqi[mask],seqj[mask],ridx[mask]
            deltas = deltas[:,ridx,ai,aj]

            # hbond correction
            use_hb_dis = (
                ljcorr[seqi,ai,0]*ljcorr[seqj,aj,1]
                + ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] ).nonzero()
            use_ohdon_dis = ( # OH are both donors & acceptors
                ljcorr[seqi,ai,0]*ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0]
                +ljcorr[seqi,ai,0]*ljcorr[seqj,aj,0]*ljcorr[seqj,aj,1]
            ).nonzero()
            use_hb_hdis = (
                ljcorr[seqi,ai,2]*ljcorr[seqj,aj,1]
                +ljcorr[seqi,ai,1]*ljcorr[seqj,aj,2]
            ).nonzero()

            # disulfide correction
            potential_disulf = (ljcorr[seqi,ai,3]*ljcorr[seqj,aj,3] ).nonzero()

            ljrs = ljparams[seqi,ai,0] + ljparams[seqj,aj,0]
            ljrs[use_hb_dis] = lj_hb_dis
            ljrs[use_ohdon_dis] = lj_OHdon_dis
            ljrs[use_hb_hdis] = lj_hbond_hdis

            ljss = torch.sqrt( ljparams[seqi,ai,1] * ljparams[seqj,aj,1] + eps )
            ljss [potential_disulf] = 0.0

            natoms = torch.sum(aamask[seq])
            ljval_i,dljEdd_i = LJLoss.ljVdV(deltas,ljrs,ljss,lj_lin,eps)

            ljval += ljval_i / natoms

            # sum per-atom-pair grads into per-atom grads
            # note this is stochastic op on GPU
            idxI,idxJ = rii[ridx]*A + ai, rjj[ridx]*A + aj

            dljEdx.view(N,-1,3).index_add_(1, idxI, dljEdd_i[...,None]*deltas, alpha=1.0/natoms)
            dljEdx.view(N,-1,3).index_add_(1, idxJ, dljEdd_i[...,None]*deltas, alpha=-1.0/natoms)

        ctx.save_for_backward(dljEdx)

        return ljval



    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        dljEdx, = ctx.saved_tensors
        return (
            grad_output * dljEdx,
            None, None, None, None, None, None, None, None, None, None, None, None, None
        )

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds,
    lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75,
    lj_maxrad=-1.0, eps=1e-8,
    training=True
):
    lj = LJLoss.apply
    ljval = lj(
        xs, seq, aamask, bond_feats, dist_matrix, ljparams, ljcorr, num_bonds,
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training)

    return ljval


def calc_chiral_loss(pred, chirals):
    """
    calculate error in dihedral angles for chiral atoms
    Input:
     - pred: predicted coords (B, L_ligand, 3)
     - chirals: True coords (B, nchiral, 5), skip if 0 chiral sites, 5 dimension are indices for 4 atoms that make dihedral and the ideal angle they should form
    Output:
     - mean squared error of chiral angles
    """
    if chirals.shape[1] == 0:
        return torch.tensor(0.0, device=pred.device)
    chiral_dih = pred[:, chirals[..., :-1].long()]
    pred_dih = get_dih(chiral_dih[..., 0, :], chiral_dih[..., 1, :], chiral_dih[..., 2, :], chiral_dih[..., 3, :])  # n_symm, b, n, 36, 3
    l = torch.square(pred_dih - chirals[..., -1]).mean()
    return l

@torch.enable_grad()
def calc_lj_grads(
    seq, xyz, alpha, toaa, bond_feats, dist_matrix,
    aamask, ljparams, ljcorr, num_bonds,
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75,
    lj_maxrad=-1.0, eps=1e-8
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Elj = calc_lj(
        seq[0],
        xyzaa[...,:3],
        aamask,
        bond_feats,
        dist_matrix,
        ljparams,
        ljcorr,
        num_bonds,
        lj_lin,
        lj_hb_dis,
        lj_OHdon_dis,
        lj_hbond_hdis,
        lj_maxrad,
        eps
    )
    return torch.autograd.grad(Elj, (xyz,alpha))

@torch.enable_grad()
def calc_chiral_grads(xyz, chirals):
    xyz.requires_grad_(True)
    l = calc_chiral_loss(xyz, chirals)
    if l.item() == 0.0:
        return (torch.zeros(xyz.shape, device=xyz.device),) # autograd returns a tuple..
    return torch.autograd.grad(l, xyz)


# 解析旋转等价的侧链
def resolve_symmetry(xs, Rsnat_all, xsnat, Rsnat_all_alt, xsnat_alt, atm_mask):
    dists = torch.linalg.norm(xs[:, :, None, :] - xs[atm_mask, :][None, None, :, :], dim=-1)
    dists_nat = torch.linalg.norm(xsnat[:, :, None, :] - xsnat[atm_mask, :][None, None, :, :], dim=-1)
    dists_natalt = torch.linalg.norm(xsnat_alt[:, :, None, :] - xsnat_alt[atm_mask, :][None, None, :, :], dim=-1)

    drms_nat = torch.sum(torch.abs(dists_nat - dists), dim=(-1, -2))
    drms_natalt = torch.sum(torch.abs(dists - dists_natalt), dim=(-1, -2))

    Rsnat_symm = Rsnat_all
    xs_symm = xsnat

    toflip = drms_natalt < drms_nat

    Rsnat_symm[toflip, ...] = Rsnat_all_alt[toflip, ...]
    xs_symm[toflip, ...] = xsnat_alt[toflip, ...]

    return Rsnat_symm, xs_symm


def focal_loss_multiclass(logits, targets, gamma=2.0, alpha=None, reduction='mean'):
    """
    logits:  (N, K)
    targets: (N,)
    alpha:   (K,) or None
    """
    log_probs = F.log_softmax(logits, dim=-1)       # (N, K)
    probs = torch.exp(log_probs)

    # 取真实类别的 log_prob
    log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    focal_term = (1 - pt) ** gamma
    loss = - focal_term * log_pt

    if alpha is not None:
        alpha_t = alpha[targets]
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def torsionAngleLoss(alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8):
    I = alpha.shape[0]
    lnat = torch.sqrt(torch.sum(torch.square(alpha), dim=-1) + eps)
    anorm = alpha / (lnat[..., None])

    l_tors_ij = torch.min(
        torch.sum(torch.square(anorm - alphanat[None]), dim=-1),
        torch.sum(torch.square(anorm - alphanat_alt[None]), dim=-1)
    )

    l_tors = torch.sum(l_tors_ij * tors_mask[None]) / (torch.sum(tors_mask) * I + eps)
    l_norm = torch.sum(torch.abs(lnat - 1.0) * tors_mask[None]) / (torch.sum(tors_mask) * I + eps)
    l_planar = torch.sum(torch.abs(alpha[..., 0]) * tors_planar[None]) / (torch.sum(tors_planar) * I + eps)

    return l_tors + 0.02 * l_norm + 0.02 * l_planar


def frame_distance_loss(R_pred, T_pred, R_true, T_true, mask, w_trans=0.5, w_rots=1.0, d_clamp=10., p_clamp=0.9, gamma=1.0, eps=1e-8):
    """
    计算蛋白质框架损失，使用RFDiffusion中的计算方式
    """

    I, B, L = R_pred.shape[:3]
    if L == 0:
        print('WARNING: frame_distance_loss called with L=0. Returning 0.0 loss.')
        return torch.tensor(0.0).to(R_pred.device)

    R_true = R_true.unsqueeze(0)  # (1, B, L, 3, 3)
    T_true = T_true.unsqueeze(0)  # (1, B, L, 3)
    mask = mask.unsqueeze(0)  # (1, B, L)

    """平移向量部分损失"""
    trans_err = torch.norm(T_pred - T_true, dim=-1)  # (I, B, L)
    if torch.rand(1).item() < p_clamp:  # 概率时间截断至最大距离
        trans_err = torch.clamp(trans_err, max=d_clamp) ** 2
    else:
        trans_err = trans_err ** 2

    """旋转矩阵部分损失"""
    R_pred_inv = R_pred.transpose(-1, -2)
    R_diff = torch.matmul(R_pred_inv, R_true)  # (I, B, L, 3, 3)
    I_3 = torch.eye(3, device=R_pred.device).view(1, 1, 1, 3, 3)  # 构造单位矩阵
    rot_err_matrix = I_3 - R_diff  # 计算误差矩阵
    # 计算Frobenius范数的平方（即矩阵内所有元素的平方和）
    rots_err = torch.sum(rot_err_matrix ** 2, dim=(-1, -2))  # (I, B, L)

    frame_loss = w_trans * trans_err + w_rots * rots_err  # (I, B, L)
    frame_loss = torch.sum(frame_loss * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)  # (I, B)

    trans_err = torch.sum(trans_err * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)  # (I, B)
    rots_err = torch.sum(rots_err * mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)  # (I, B)

    # decay on loss over iterations
    w_loss = torch.pow(torch.full((I,), gamma, device=R_pred.device), torch.arange(I, device=R_pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    # 按照RFdiffusion的框架损失定义，应用权重前要进行平方
    frame_loss = frame_loss * w_loss[:, None]
    trans_err = trans_err * w_loss[:, None]
    rots_err = rots_err * w_loss[:, None]

    return frame_loss.sum() / B, trans_err.sum() / B, rots_err.sum() / B


def calc_ligand_coord_loss(pred_xyz_stack, true_xyz, ligand_mask, gamma=1.0, eps=1e-8):
    """
    计算配体的坐标损失，解决对称性问题。
    pred_xyz_stack: (I, B, L, 3)  模型输出的堆叠预测坐标
    true_xyz:       (B, L, 3)     真实的 Holo 坐标
    ligand_mask:    (B, L)        Boolean mask, True 表示配体原子
    """
    I = pred_xyz_stack.shape[0]

    # coords_err = torch.sum((true_xyz[None] - pred_xyz_stack) ** 2 * ligand_mask[None, :, :, None], dim=(-1, -2)) / (ligand_mask.sum(dim=-1) + eps)
    # 使用Huber loss计算损失，主要用途是抑制方差过大导致梯度过大，从而导致模型优化困难
    coords_err = F.huber_loss(pred_xyz_stack, true_xyz[None], delta=2., reduction='none')
    coords_err = coords_err.mean(dim=(1, 2, 3))

    # 获取每个预测结果的损失权重
    w_loss = torch.pow(torch.full((I,), gamma, device=pred_xyz_stack.device), torch.arange(I, device=pred_xyz_stack.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    total_loss = (w_loss * coords_err).sum()  # 应用权重

    return total_loss.mean()


def calc_c6d_loss(logit_s, label_s, mask_2d, eps=1e-8):
    loss_s = list()
    for i in range(len(logit_s)):
        loss = nn.CrossEntropyLoss(reduction='none')(logit_s[i], label_s[..., i])  # (B, L, L)
        loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
        loss_s.append(loss)
    loss_s = torch.stack(loss_s)
    return loss_s.sum()


# fd allatom lddt
def calc_allatom_lddt_loss(P, Q, pred_lddt, idx, atm_mask, mask_2d, same_chain, pocket_mask=None, negative=False, interface=False, bin_scaling=1, N_stripe=1, eps=1e-8):
    # https://github.com/RosettaCommons/RFDpoly/blob/main/rf_diffusion/RF2-allatom/rf2aa/loss.py#L1572
    # P - N x L x natoms x 3
    # Q - L x natoms x 3
    # pred_lddt - 1 x nbucket x L
    # idx - 1 x L
    #
    N, L, Natm = P.shape[:3]

    plddt = lddt_unbin(pred_lddt)[0]

    # striped evaluation of L x L x N_atoms x N_atoms distances to save GPU mem
    L_stripe = int(np.ceil(L / N_stripe))  # how many residues in each stripe
    lddt_s = []
    pair_mask_accum = torch.zeros((N, L, Natm), device=P.device)
    for i1 in np.arange(0, L, L_stripe):
        i2 = min(i1 + L_stripe, L)

        # distance matrix
        Pij = torch.square(P[:, i1:i2, None, :, None, :] - P[:, None, :, None, :, :])  # (N, L_stripe, L, 27, 27)
        Pij = torch.sqrt(Pij.sum(dim=-1) + eps)
        Qij = torch.square(Q[None, i1:i2, None, :, None, :] - Q[None, None, :, None, :, :])  # (1, L_stripe, L, 27, 27)
        Qij = torch.sqrt(Qij.sum(dim=-1) + eps)

        # get valid pairs
        pair_mask = torch.logical_and(Qij > 0, Qij < 15).float()  # only consider atom pairs within 15A
        # ignore missing atoms
        pair_mask *= (atm_mask[:, i1:i2, None, :, None] * atm_mask[:, None, :, None, :]).float()

        # ignore atoms within same residue
        pair_mask *= (idx[:, i1:i2, None, None, None] != idx[:, None, :, None, None]).float()  # (1, L_stripe, L, 27, 27)
        if negative:
            pair_mask *= same_chain.bool()[:, i1:i2, :, None, None]  # 忽略不同链之间的原子
        elif interface:
            pair_mask *= ~same_chain.bool()[:, i1:i2, :, None, None]  # 忽略相同链之间的原子

        pair_mask *= mask_2d.bool()[:, i1:i2, :, None, None]

        delta_PQ = torch.abs(Pij - Qij + eps)  # (N, L_stripe, L, 14, 14)

        lddt_ = torch.zeros((N, i2 - i1, Natm), device=P.device)
        for distbin in (0.5, 1.0, 2.0, 4.0):
            lddt_ += 0.25 * torch.sum((delta_PQ <= distbin * bin_scaling) * pair_mask, dim=(2, 4)) / (torch.sum(pair_mask, dim=(2, 4)) + eps)
        lddt_s.append(lddt_)
        pair_mask_accum += pair_mask.sum(dim=(1, 3))

    lddt = torch.cat(lddt_s, dim=1)  # (N, L, Natm)

    final_lddt_by_res = torch.clamp((lddt[-1] * atm_mask[0]).sum(-1) / (atm_mask.sum(-1) + eps), min=0.0, max=1.0)

    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    true_lddt_label = torch.bucketize(final_lddt_by_res[None, ...], lddt_bins).long()
    lddt_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred_lddt, true_lddt_label[-1])

    res_mask = atm_mask.any(dim=-1)
    lddt_loss = (lddt_loss * res_mask).sum() / (res_mask.sum() + eps)

    # method 1: average per-residue
    # lddt = lddt.sum(dim=-1) / (atm_mask.sum(dim=-1)+1e-8) # L
    # lddt = (res_mask*lddt).sum() / (res_mask.sum() + 1e-8)

    # method 2: average per-atom
    atm_mask = atm_mask * (pair_mask_accum != 0)
    all_lddt = (lddt * atm_mask).sum(dim=(1, 2)) / (atm_mask.sum() + eps)
    if pocket_mask is not None:
        pocket_lddt = (lddt * atm_mask * pocket_mask[..., None]).sum(dim=(1, 2)) / ((atm_mask * pocket_mask[..., None]).sum() + eps)
        return all_lddt.mean(), pocket_lddt.mean(), plddt.mean(), lddt_loss
    else:
        return all_lddt.mean(), plddt.mean(), lddt_loss


# use improved coordinate frame generation
def get_t(N, Ca, C, eps=1e-8, non_ideal=False):
    I, B, L = N.shape[:3]
    Rs, Ts = rigid_from_3_points(N.view(I * B, L, 3), Ca.view(I * B, L, 3), C.view(I * B, L, 3), non_ideal=non_ideal, eps=eps)
    Rs = Rs.view(I, B, L, 3, 3)
    Ts = Ts.view(I, B, L, 3)
    t = Ts[:, :, None] - Ts[:, :, :, None]  # t[0,1] = residue 0 -> residue 1 vector
    return torch.einsum('iblkj, iblmk -> iblmj', Rs, t)  # (I,B,L,L,3)


def calc_bb_fape_loss(pred, true, mask_2d, same_chain, d_clamp=10.0, p_clamp=0.9, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-8):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]

    true = true.unsqueeze(0)
    # 计算真实结构和预测结构的相对位移向量
    t_tilde_ij = get_t(true[:, :, :, 0], true[:, :, :, 1], true[:, :, :, 2], non_ideal=True)
    t_ij = get_t(pred[:, :, :, 0], pred[:, :, :, 1], pred[:, :, :, 2])
    # 计算相对位移向量的差异
    difference = torch.sqrt(torch.square(t_tilde_ij - t_ij).sum(dim=-1) + eps)

    if d_clamp is not None and torch.rand(1).item() < p_clamp:  # 依概率p_clamp决定是否应用截断
        clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
        clamp = clamp[None]
        difference = torch.clamp(difference, max=clamp)
    loss = torch.nan_to_num(difference / A)  # (I, B, L, L)
    loss = (mask_2d[None] * loss).sum(dim=(1, 2, 3)) / (mask_2d.sum() + eps)
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()
    tot_loss = (w_loss * loss).sum()
    return tot_loss


# from Ivan: FAPE generalized over atom sets & frames
def compute_general_FAPE(X, Y, atom_mask, frames, frame_mask, frame_atom_mask=None, Z=10.0, d_clamp=10.0, gamma=1., eps=1e-8):
    # X (predicted) N x L x natoms x 3
    # Y (native)    1 x L x natoms x 3
    # atom_mask     1 x L x natoms
    # frames        1 x L x nframes x 3 x 2
    # frame_mask    1 x L x nframes
    # frame_atom_mask     1 x L x natoms

    if frame_atom_mask is None:
        frame_atom_mask = atom_mask

    I, L, natoms, _ = X.shape

    # flatten middle dims so can gather across residues
    X_prime = X.reshape(I, L * natoms, -1, 3).repeat(1, 1, NFRAMES, 1)
    Y_prime = Y.reshape(1, L * natoms, -1, 3).repeat(1, 1, NFRAMES, 1)

    # reindex frames for flat X
    frames_reindex = torch.zeros(frames.shape[:-1], device=frames.device)
    for i in range(L):
        frames_reindex[:, i, :, :] = (i + frames[..., i, :, :, 0]) * natoms + frames[..., i, :, :, 1]
    frames_reindex = frames_reindex.long()

    frame_mask *= torch.all(
        torch.gather(frame_atom_mask.reshape(1, L * natoms), 1, frames_reindex.reshape(1, L * NFRAMES * 3)).reshape(1, L, -1, 3),
        axis=-1)

    X_x = torch.gather(X_prime, 1, frames_reindex[..., 0:1].repeat(I, 1, 1, 3))
    X_y = torch.gather(X_prime, 1, frames_reindex[..., 1:2].repeat(I, 1, 1, 3))
    X_z = torch.gather(X_prime, 1, frames_reindex[..., 2:3].repeat(I, 1, 1, 3))
    uX, tX = rigid_from_3_points(X_x, X_y, X_z)

    Y_x = torch.gather(Y_prime, 1, frames_reindex[..., 0:1].repeat(1, 1, 1, 3))
    Y_y = torch.gather(Y_prime, 1, frames_reindex[..., 1:2].repeat(1, 1, 1, 3))
    Y_z = torch.gather(Y_prime, 1, frames_reindex[..., 2:3].repeat(1, 1, 1, 3))
    uY, tY = rigid_from_3_points(Y_x, Y_y, Y_z, non_ideal=True)

    xij = torch.einsum(
        'brji,brsj->brsi',
        uX[:, frame_mask[0]], X[:, atom_mask[0]][:, None, ...] - X_y[:, frame_mask[0]][:, :, None, ...]
    )
    xij_t = torch.einsum('rji,rsj->rsi', uY[frame_mask], Y[atom_mask][None, ...] - Y_y[frame_mask][:, None, ...])
    diff = torch.sqrt(torch.sum(torch.square(xij - xij_t[None, ...]), dim=-1) + eps)
    loss = (1.0 / Z) * (torch.clamp(diff, max=d_clamp)).mean(dim=(1, 2))  # 距离截断

    w_loss = torch.pow(torch.full((I,), gamma, device=X.device), torch.arange(I, device=X.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()
    loss = (w_loss * loss).sum()

    # pae_loss = compute_pae_loss(X, X_y, uX, Y, Y_y, uY, logit_pae) if logit_pae is not None \
    #     else torch.tensor(0).to(frames.device)
    # pde_loss = compute_pde_loss(X, Y, logit_pde) if logit_pde is not None \
    #     else torch.tensor(0).to(frames.device)
    return loss


def calc_atom_bond_loss(pred, true, bond_feats, seq, beta=0.2, eps=1e-6):
    """
    loss on distances between bonded atoms
    """
    loss_func_sum = torch.nn.SmoothL1Loss(reduction='sum', beta=beta)
    loss_func_mean = torch.nn.SmoothL1Loss(reduction='mean', beta=beta)

    # 配体内键
    atom_bonds = (bond_feats > 0) * (bond_feats < 5)  # 只考虑有意义的化学键
    b, i, j = torch.where(atom_bonds > 0)
    nat_dist = torch.sum(torch.square(true[:, i, 1] - true[:, j, 1]), dim=-1)
    pred_dist = torch.sum(torch.square(pred[:, i, 1] - pred[:, j, 1]), dim=-1)
    # lig_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist), max=clamp)) # from EquiBind
    lig_dist_loss = loss_func_sum(nat_dist, pred_dist)

    # 蛋白质残基与配体原子之间的键（即原子化残基）
    inter_bonds = bond_feats == 6
    _, i, j = torch.where(inter_bonds)
    a = (seq[:, i] < 22) & (seq[:, j] == 29)  # res N - atom C: binary indicator
    b = (seq[:, i] < 22) & (seq[:, j] == 45)  # res C - atom N
    c = (seq[:, i] == 29) & (seq[:, j] < 22)  # atom C - res N
    d = (seq[:, i] == 45) & (seq[:, j] < 22)  # atom N - res C
    i_atom = 0 * a + 2 * b + 1 * c + 1 * d  # (B, N_bonds) : indexes of atom that is bonded (N:0, C:2, 1:ligand atom)
    j_atom = 1 * a + 1 * b + 0 * c + 2 * d  # (B, N_bonds)
    nat_dist = torch.sum(torch.square(true[0, i, i_atom[0], :] - true[0, j, j_atom[0], :]), dim=-1)  # assumes B=1
    pred_dist = torch.sum(torch.square(pred[0, i, i_atom[0], :] - pred[0, j, j_atom[0], :]), dim=-1)
    # inter_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist), max=clamp))
    inter_dist_loss = loss_func_sum(nat_dist, pred_dist)

    # 直接成键损失，包括配体内部的化学键，以及蛋白质与配体之间的共价键/强相互作用。
    bond_dist_loss = (lig_dist_loss + inter_dist_loss) / (atom_bonds.sum() + inter_bonds.sum() + eps)

    # 强制执行原子间相距两个键的LAS约束以及芳香族基团的约束
    atom_bonds_np = atom_bonds[0].cpu().numpy()
    G = nx.from_numpy_array(atom_bonds_np)
    paths = find_all_paths_of_length_n(G, 2)
    # 跨键损失，相隔两个化学键的原子（即 1-3 相互作用），用于约束键角。
    if paths:
        paths = torch.tensor(paths, device=pred.device)
        nat_dist = torch.sum(torch.square(true[:, paths[:, 0], 1] - true[:, paths[:, 2], 1]), dim=-1)
        pred_dist = torch.sum(torch.square(pred[:, paths[:, 0], 1] - pred[:, paths[:, 2], 1]), dim=-1)
        # skip_bond_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist),max=clamp))/(paths.shape[0]+eps)
        skip_bond_dist_loss = loss_func_mean(nat_dist, pred_dist)
    else:
        skip_bond_dist_loss = torch.tensor(0, device=pred.device)
    rigid_groups = find_all_rigid_groups(bond_feats)
    # 刚性基团损失，保持芳香环等刚性基团的平面性和形状。
    if rigid_groups != None:
        nat_dist = torch.sum(torch.square(true[:, rigid_groups[:, 0], 1] - true[:, rigid_groups[:, 1], 1]), dim=-1)
        pred_dist = torch.sum(torch.square(pred[:, rigid_groups[:, 0], 1] - pred[:, rigid_groups[:, 1], 1]), dim=-1)
        # rigid_group_dist_loss = torch.sum(torch.clamp(torch.square(nat_dist-pred_dist),max=clamp))/(rigid_groups.shape[0]+eps)
        rigid_group_dist_loss = loss_func_mean(nat_dist, pred_dist)
    else:
        rigid_group_dist_loss = torch.tensor(0, device=pred.device)

    return bond_dist_loss, skip_bond_dist_loss, rigid_group_dist_loss


def calc_pae_loss(X, X_y, uX, Y, Y_y, uY, logit_pae, pae_bin_step=0.5, eps=1e-4):
    # predicted aligned error: C-alpha (or sm. mol atom) distances in backbone frames
    xij_ca = torch.einsum('rji,rsj->rsi', uX[-1,:,0], X[-1,:,None,1] - X_y[-1,None,:,0,:]) # last bb prediction
    xij_ca_t = torch.einsum('rji,rsj->rsi', uY[0,:,0], Y[0,:,None,1] - Y_y[0,None,:,0,:]) # assumes B=1
    eij_label = torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1)+eps).clone().detach()

    nbin = logit_pae.shape[1]
    pae_bins = torch.linspace(pae_bin_step, pae_bin_step*(nbin-1), nbin-1, dtype=logit_pae.dtype, device=logit_pae.device)
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
    return torch.nn.CrossEntropyLoss(reduction='mean')(logit_pae, true_pae_label[None]) # assumes B=1


# ideal N-C distance, ideal cos(CA-C-N angle), ideal cos(C-N-CA angle)
# for NA, we do not compute this as it is not computable from the stubs alone
def calc_BB_bond_geom(pred, idx, prot_BB_mask, eps=1e-8, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, sig_len=0.02, sig_ang=0.05):
    '''
    计算主链键几何参数（键长和键角）以及损失值。
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''

    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    def length(a, b):
        return torch.norm(a - b, dim=-1)

    B, L = pred.shape[:2]

    bonded = (idx[:, 1:] - idx[:, :-1]) == 1
    prot_BB_mask = prot_BB_mask[:, :-1]

    # bond length: C-N
    blen_CN_pred = length(pred[:, :-1, 2], pred[:, 1:, 0]).reshape(B, L - 1)  # (B, L-1)
    CN_loss = torch.clamp(torch.abs(blen_CN_pred - ideal_NC) - sig_len, min=0.0)
    CN_loss = (bonded * prot_BB_mask * CN_loss).sum() / ((bonded * prot_BB_mask).sum() + eps)
    blen_loss = CN_loss  # fd squared loss

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(B, L - 1)
    bang_CNCA_pred = cosangle(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(B, L - 1)
    CACN_loss = torch.clamp(torch.abs(bang_CACN_pred - ideal_CACN) - sig_ang, min=0.0)
    CACN_loss = (bonded * prot_BB_mask * CACN_loss).sum() / ((bonded * prot_BB_mask).sum() + eps)
    CNCA_loss = torch.clamp(torch.abs(bang_CNCA_pred - ideal_CNCA) - sig_ang, min=0.0)
    CNCA_loss = (bonded * prot_BB_mask * CNCA_loss).sum() / ((bonded * prot_BB_mask).sum() + eps)
    bang_loss = CACN_loss + CNCA_loss

    return blen_loss, bang_loss
