import torch
import torch.nn as nn
import model.utils.torus as torus
import model.utils.so3_utils as so3_utils
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import scatter
from torch_scatter import scatter_add, scatter
from .so3.dist import uniform_so3
from .chemical import NBTYPES

class Interpolant(nn.Module):
    """
    对蛋白质复合物相关数据进行插值。主要是构建从apo到holo的条件概率路径
    构建的路径包括：apo到holo的SE(3)、侧链扭转角路径；小分子配体的从先验分布（高斯）到真实配体的路径
    """

    def __init__(self, T=1., trans_scale=10., min_t=1e-3,
                 seq_noise_scale=1.0, seq_max_temp=10., decay=3.,
                 atom_noise_scale=0.8, atom_max_temp=8.0, atom_decay=4.0,
                 bond_noise_scale=0.5, bond_max_temp=3.0, bond_decay=5.0,
                 device='cpu', **akwags):
        super(Interpolant, self).__init__()

        self.T = T

        self.trans_scale = trans_scale
        self.min_t = min_t

        self.seq_noise_scale = seq_noise_scale
        self.seq_max_temp = seq_max_temp
        self.decay = decay
        # 原子类型和化学键类型的离散分布不同，因此使用独立的加噪与退火参数。
        self.atom_noise_scale = atom_noise_scale
        self.atom_max_temp = atom_max_temp
        self.atom_decay = atom_decay
        self.bond_noise_scale = bond_noise_scale
        self.bond_max_temp = bond_max_temp
        self.bond_decay = bond_decay

        self.device = device

        self.to_numpy = lambda x: x.detach().cpu().numpy()


    def _sample_t(self, batch_size):
        """
        批量随机时间步长采样
        """
        eps_t = torch.rand(batch_size, device=self.device)
        # 反向采样：通过将每个样本与互补样本配对来降低方差
        offset = torch.arange(batch_size, device=self.device) / batch_size  # 计算采样时间步之间的间隔
        # 确保每个采样时间步在[0, 1)区间内均匀分布，提高训练稳定性
        eps_t = ((eps_t / batch_size) + offset) % 1
        # 确保值不是精确的0或1，并使用配置中的最小时间步。
        t = (1 - self.min_t) * eps_t + self.min_t
        return t

    def _zero_center_part(self, pos, gen_mask, res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[..., None], dim=1) / (torch.sum(gen_mask, dim=-1, keepdim=True) + 1e-8)  # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1)  # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[..., None]
        return pos, center

    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(self.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _interpolate_trans(self, trans_1, t, trans_0=None, mask=None, res_mask=None):
        """
        对平移向量进行插值
        """
        if trans_0 is None:
            trans_0 = torch.randn_like(trans_1, device=trans_1.device) * self.trans_scale
            trans_0, _ = self._zero_center_part(trans_0, mask, res_mask)
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = torch.where(mask[..., None], trans_t, trans_1)
        return trans_t

    def _interpolate_rots(self, rots_1, t, rots_0=None, mask=None):
        """
        对旋转进行插值
        """
        B, L = rots_1.shape[:2]
        if rots_0 is None:
            rots_0 = uniform_so3(B, L, device=rots_1.device)
        rots_t = so3_utils.geodesic_t(t[..., None], rots_1, rots_0)
        rots_t = torch.where(mask[..., None, None], rots_t, rots_1)
        return rots_t

    def _interpolate_chi_angles(self, chi_1, t, chi_0=None, mask=None):
        """
        对侧链扭转角进行插值。
        chi_0：apo结构的扭转角sin/cos
        chi_1：holo结构的扭转角sin/cos
        """
        # 转换为角度，如果没输入chi_0（apo）则随机初始化
        if chi_0 is None:
            chi_0 = torus.tor_random_uniform(chi_1.shape[:3], device=chi_1.device, dtype=chi_1.dtype)  # (B,L,10)
        else:
            chi_0 = torch.atan2(chi_0[:, :, :, 0], chi_0[:, :, :, 1])  # atan2输出的角度在[-π, π]
        chi_1 = torch.atan2(chi_1[:, :, :, 0], chi_1[:, :, :, 1])

        # 计算最短路径角速度：(diff + pi) % 2pi - pi，目的是为了限定在单位圆中
        delta_theta = torch.remainder(chi_1 - chi_0 + torch.pi, 2 * torch.pi) - torch.pi
        # 插值得到当前角度
        theta_t = chi_0 + t.squeeze() * delta_theta
        sin_t = torch.sin(theta_t)
        cos_t = torch.cos(theta_t)
        x_t = torch.stack([sin_t, cos_t, mask.to(dtype=cos_t.dtype)], dim=-1)  # 转换回sin/cos
        return x_t

    def _interpolate_ligand_coords(self, ligand_coords_1, t, mask=None, is_atomize_protein_mask=None, ligand_coords_0=None):
        """
        对配体坐标进行插值
        """
        B, L = ligand_coords_0.shape[:2]
        if mask is None:
            mask = torch.ones((B, L), device=ligand_coords_1.device)
        # TODO: 由于只有一个样本，只会使用Kabsch进行结构对齐，在匈牙利算法匹配中会失去作用。需要探讨使用大于1的批次的使用流程
        if is_atomize_protein_mask.sum().item() > 0:
            assert ligand_coords_0 is not None, '当含有原子化后的残基时，原子化后的残基的部分插值必须根据apo来（即需要显示传入ligand_coords_0）'
            # 原子化后的残基使用apo部分作为初始状态
            ligand_coords_0 = torch.concat([ligand_coords_0[is_atomize_protein_mask].view(B, -1, 3),
                                            self._batch_ot(torch.randn_like(ligand_coords_1[~is_atomize_protein_mask].view(B, -1, 3), device=ligand_coords_1.device) * self.trans_scale,
                                                           ligand_coords_1[~is_atomize_protein_mask].view(B, -1, 3), mask[~is_atomize_protein_mask].view(B, -1))], dim=1)
        else:  # 无共价键，正常插值
            ligand_coords_0 = torch.randn_like(ligand_coords_1, device=ligand_coords_1.device) * self.trans_scale
            ligand_coords_0 = self._batch_ot(ligand_coords_0, ligand_coords_1, mask)  # 使用算法寻找噪声的最近匹配
        ligand_coords_t = (1 - t[..., None]) * ligand_coords_0 + t[..., None] * ligand_coords_1
        return ligand_coords_t

    def _interpolate_seq(self, seq1hot, t, mask=None, noise_scale=None, max_temp=None, decay=None, eps=1e-20):
        def _onehot_to_clamped(seq_one_hot, eps=1e-5):
            return torch.clamp(seq_one_hot, min=eps, max=1.0)

        def _temperature_function(expanded_t, decay_rates=None):
            if decay_rates is not None:
                assert decay_rates.shape == expanded_t.shape, "shape mismatch"
                return max_temp * torch.exp(-decay_rates * expanded_t)
            return max_temp * torch.exp(-decay * expanded_t)

        """
        根据《Gumbel-Softmax Flow Matching with Straight-Through Guidance for Controllable Biological Sequence Generation》提到的方法给离散token插值。
        根据时间步 t 注入随机噪声，并转化为一种“模糊”的概率分布状态。
        """
        noise_scale = self.seq_noise_scale if noise_scale is None else noise_scale
        max_temp = self.seq_max_temp if max_temp is None else max_temp
        decay = self.decay if decay is None else decay
        B, L, V = seq1hot.shape

        expanded_t = t.unsqueeze(-1).unsqueeze(-1).expand(B, L, V)  # 插值时间步
        logits = torch.log(_onehot_to_clamped(seq1hot.float()))  # 将概率转为log-prob，避免Gumbel噪声淹没真实类别
        # Gumbel噪声生成。Gumbel噪声是专门为离散分类分布设计的
        rand_eps = max(eps, torch.finfo(logits.dtype).eps)
        U = torch.rand_like(logits).clamp(min=rand_eps, max=1.0 - rand_eps)
        gumbel_noise = -torch.log(-torch.log(U))
        gumbel_noise = noise_scale * gumbel_noise
        temp = _temperature_function(expanded_t)  # 温度随时间步变化，用以控制噪声强度

        xt = (logits + gumbel_noise) / temp  # 添加噪声并缩放
        xt = torch.softmax(xt, dim=-1)  # 归一化为概率分布

        if mask is not None:
            xt = torch.where(mask[..., None], xt, seq1hot)  # 还原无需生成的部分，对于共价结合的原子化的残基部分，其不会被插值
        xt_seq = xt.argmax(dim=-1)

        return xt, xt_seq, temp

    def forward(self, batch, t=None):
        seq_batch = batch['seq']
        xyzs_batch = batch['xyzs']

        B, L, _ = seq_batch['seq1hot'].shape

        if t is None:
            t = self._sample_t(B)  # 采样时间步。必须保证所有的数据插值的时间步一致。
        else:
            t = torch.tensor(t, device=seq_batch['seq1hot'].device).repeat(B)

        # 对平移向量进行插值
        trans_t = self._interpolate_trans(trans_1=xyzs_batch['xyzs_true_trans'], trans_0=xyzs_batch['xyzs_prev_trans'], t=t, mask=xyzs_batch['res_mask'])

        # 对旋转矩阵进行插值
        rots_t = self._interpolate_rots(rots_1=xyzs_batch['xyzs_true_rots'], rots_0=xyzs_batch['xyzs_prev_rots'], t=t, mask=xyzs_batch['res_mask'])

        # 扭转角插值
        chi_t = self._interpolate_chi_angles(chi_0=xyzs_batch['torsion_angle_prev'], chi_1=xyzs_batch['torsion_angle_true'],
                                             t=t, mask=xyzs_batch['torsion_mask'])

        # 配体部分坐标插值。原子化残基的初始状态会不一样
        if batch['is_atomize_protein'].sum().item() > 0:
            ligand_xyzs_1 = torch.concat([xyzs_batch['xyzs_true'][batch['is_atomize_protein']][:, 1].reshape(B, -1, 3),
                                          xyzs_batch['xyzs_true'][~batch['is_protein']][:, 1].reshape(B, -1, 3)], dim=1)  # (B, L_l, 3)
            ligand_xyzs_0 = torch.concat([xyzs_batch['xyzs_prev'][batch['is_atomize_protein']][:, 1].reshape(B, -1, 3),
                                          xyzs_batch['xyzs_prev'][~batch['is_protein']][:, 1].reshape(B, -1, 3)], dim=1)  # (B, L_l, 3)
            # TODO: 目前默认传入的batch=1
            is_atomize_protein_mask = torch.concat([batch['is_atomize_protein'][batch['is_atomize_protein']].reshape(B, -1),
                                                    batch['is_protein'][~batch['is_protein']].reshape(B, -1)], dim=-1)
        else:
            ligand_xyzs_1 = xyzs_batch['xyzs_true'][~batch['is_protein']][:, 1].reshape(B, -1, 3)  # (B, L_l, 3)
            ligand_xyzs_0 = xyzs_batch['xyzs_prev'][~batch['is_protein']][:, 1].reshape(B, -1, 3)  # (B, L_l, 3)
            # TODO: 目前默认传入的batch=1
            is_atomize_protein_mask = batch['is_protein'][~batch['is_protein']].reshape(B, -1)
        ligand_xyzs_t = self._interpolate_ligand_coords(ligand_coords_1=ligand_xyzs_1, t=t, ligand_coords_0=ligand_xyzs_0,
                                                        is_atomize_protein_mask=is_atomize_protein_mask)

        # token还原操作包括原子化残基部分
        token_mask = torch.where(batch['is_atomize_protein'], True, batch['is_protein'])

        # 序列部分插值。需要注意蛋白质部分的token无需插值
        xt, xt_seq, temp = self._interpolate_seq(
            seq_batch['seq1hot'], t, mask=~batch['is_protein'],
            noise_scale=self.atom_noise_scale, max_temp=self.atom_max_temp, decay=self.atom_decay
        )
        xt = torch.where(token_mask[..., None], seq_batch['seq1hot'], xt)  # 将蛋白质部分还原
        xt_seq = torch.where(token_mask, seq_batch['seq'], xt_seq)

        # 化学键部分插值。需要注意蛋白质部分的键以及原子化残基的化学键无需插值
        bond_feats_onehot = nn.functional.one_hot(batch['bond_feats'], num_classes=NBTYPES)
        xt_bond_logits, xt_bond_feats, temp = self._interpolate_seq(
            bond_feats_onehot.reshape(B, L * L, -1), t,
            noise_scale=self.bond_noise_scale, max_temp=self.bond_max_temp, decay=self.bond_decay
        )
        xt_bond_feats = xt_bond_feats.reshape(B, L, L)  # [B, L, L]
        xt_bond_logits = xt_bond_logits.reshape(B, L, L, NBTYPES)  # [B, L, L, NBTYPES]
        xt_bond_feats = torch.where(token_mask[..., None] * token_mask[:, None, :], batch['bond_feats'], xt_bond_feats)  # 将蛋白质部分还原
        xt_bond_logits = torch.where((token_mask[..., None] * token_mask[:, None, :])[..., None], bond_feats_onehot, xt_bond_logits)

        return {'t': t, 'trans_t': trans_t, 'rots_t': rots_t, 'chi_t': chi_t, 'ligand_xyzs_t': ligand_xyzs_t, 'seq_xt': xt,
                'seq_token_xt': xt_seq, 'bond_xt': xt_bond_feats, 'bond_xt_logtis': xt_bond_logits}

def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64)
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots

@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
        """
        Move the molecule center to zero for sparse position tensors.

        Args:
            pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
            batch_indexes: [N] batch index for each atom in sparse batch format.

        Returns:
            pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
        """
        assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

        means = scatter(pos, batch_indexes, dim=0, reduce="mean")
        return pos - means[batch_indexes]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices
