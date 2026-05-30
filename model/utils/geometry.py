import torch
import torch.nn.functional as F
from .chemical import BBHeavyAtom


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def align(pos_1, pos_2, pos_mask):
    """(L,A,3),(L,A) 使用Kabsch算法对齐输入的结构"""
    L, A, _ = pos_1.shape
    x = torch.masked_select(pos_1, pos_mask.bool().unsqueeze(-1)).reshape(-1, 3)
    y = torch.masked_select(pos_2, pos_mask.bool().unsqueeze(-1)).reshape(-1, 3)
    xm, ym = x.mean(dim=0), y.mean(dim=0)  # (1,A,3)
    # 移动至几何中心
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    s = x.T @ y  # 协方差矩阵
    u, sigma, vt = torch.linalg.svd(s)  # 奇异值分解
    r = vt.T @ u.T  # (3,3)  # 旋转矩阵
    # 如果行列式小于 0，说明发生了一次破坏蛋白质手性的镜像翻转，必须反转 V^T 的最后一行来纠正旋转矩阵
    if torch.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = ym - r @ xm  # 平移向量
    pos_1_aligned = ((r @ pos_1.reshape(-1, 3).T).T + t).reshape(L, A, 3)  # (-1,3) -> (L,A,3)

    return pos_1_aligned, pos_2, r.float(), t.float()


def align_residue_level(pos_1, pos_2, pos_mask):
    """
    使用Kabsch算法对齐输入的结构（残基级别）
    pos_1: (L, A, 3)
    pos_2: (L, A, 3)
    pos_mask: (L, A)
    return: pos_1_aligned (L, A, 3), pos_2 (L, A, 3), r (L, 3, 3), t (L, 3)
    """
    L, A, _ = pos_1.shape
    mask = pos_mask.bool().unsqueeze(-1)

    # 计算每个残基的几何中心。使用 clamp(min=1) 避免全被 mask 的残基在除法时出现 NaN
    valid_atoms_per_res = mask.sum(dim=1).clamp(min=1)  # (L, 1)

    x_masked = pos_1.masked_fill(~mask, 0.0)
    y_masked = pos_2.masked_fill(~mask, 0.0)

    xm = x_masked.sum(dim=1) / valid_atoms_per_res  # (L, 3)
    ym = y_masked.sum(dim=1) / valid_atoms_per_res  # (L, 3)

    # 将坐标移动至各自残基的几何中心
    x_centered = (pos_1 - xm.unsqueeze(1)).masked_fill(~mask, 0.0)  # (L, A, 3)
    y_centered = (pos_2 - ym.unsqueeze(1)).masked_fill(~mask, 0.0)  # (L, A, 3)

    # 计算每个残基的协方差矩阵
    s = torch.bmm(x_centered.transpose(1, 2), y_centered)  # (L, 3, 3)

    # 批量奇异值分解 SVD
    u, sigma, vt = torch.linalg.svd(s)  # u: (L, 3, 3), vt: (L, 3, 3)

    # 计算批量旋转矩阵 r = V @ U^T。vt 是 V^T, u 是 U
    r = torch.bmm(vt.transpose(1, 2), u.transpose(1, 2))  # (L, 3, 3)

    # 处理可能破坏蛋白质手性的镜像翻转 (行列式 < 0)
    det = torch.linalg.det(r)  # (L,)
    neg_det_mask = det < 0
    if neg_det_mask.any():
        vt_clone = vt.clone()
        # 仅对行列式为负的残基，将其 V^T 的最后一行反转
        vt_clone[neg_det_mask, -1, :] *= -1.0
        r = torch.bmm(vt_clone.transpose(1, 2), u.transpose(1, 2))  # 重新计算 r

    # 7. 计算每个残基的平移向量 t = ym - r @ xm
    # xm.unsqueeze(2): (L, 3, 1), r: (L, 3, 3)
    t = ym.unsqueeze(2) - torch.bmm(r, xm.unsqueeze(2))  # (L, 3, 1)
    t = t.squeeze(2)  # (L, 3)

    # 8. 应用对齐到 pos_1
    # 批量应用 r 和 t： (L, A, 3) @ (L, 3, 3)^T + (L, 1, 3)
    pos_1_aligned = torch.bmm(pos_1, r.transpose(1, 2)) + t.unsqueeze(1)

    return pos_1_aligned, pos_2, r.float(), t.float()


def iterative_rigid_core_align(apo_ca, holo_ca, valid_mask=None, max_iter=10, distance_threshold=3.0):
    """
    使用迭代 Kabsch 算法计算刚性核心对齐矩阵。
    apo_ca: Apo 态的 C-alpha坐标，形状为 (L, 3)
    holo_ca: Holo 态的 C-alpha 坐标，形状为 (L, 3)
    valid_mask: 形状为 (L,) 的布尔张量，True 表示该残基坐标有效且参与对齐
    max_iter: 最大迭代次数
    distance_threshold: 判定为“刚性核心”的距离阈值 (埃)
    """
    L = apo_ca.shape[0]
    device = apo_ca.device

    if valid_mask is None:
        valid_mask = torch.ones(L, dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(dtype=torch.bool, device=device)

    # 初始化
    core_mask = valid_mask.clone()

    for i in range(max_iter):
        # 1. 提取当前核心区域的坐标
        P_core = apo_ca[core_mask]
        Q_core = holo_ca[core_mask]

        # 防止核心原子太少导致 SVD 崩溃 (至少需要 3 个点来确定 3D 刚体)
        if len(P_core) < 3:
            # print("Warning: Rigid core collapsed to < 3 atoms. Reverting to all atoms.")
            # 如果核心崩溃，回退到最初的有效残基集合
            core_mask = valid_mask.clone()
            P_core, Q_core = apo_ca[core_mask], holo_ca[core_mask]

        # 2. 去中心化
        t_apo = P_core.mean(dim=0)
        t_holo = Q_core.mean(dim=0)
        P_centered = P_core - t_apo
        Q_centered = Q_core - t_holo

        # 3. 计算协方差矩阵并进行 SVD 分解
        # H = P^T * Q
        H = torch.matmul(P_centered.T, Q_centered)
        U, S, V = torch.svd(H)

        # 4. 计算旋转矩阵 R = V * U^T，并处理镜像反射问题
        R = torch.matmul(V, U.T)
        if torch.det(R) < 0:
            # 如果行列式为负，说明发生了镜像翻转，反转最后一步
            V[:, 2] = -V[:, 2]
            R = torch.matmul(V, U.T)

        # 5. 将计算出的旋转平移应用到【全长】的 Apo 结构上
        # 注意：是对全长 apo_ca 进行变换，用来评估所有残基的新距离
        apo_ca_aligned = torch.matmul((apo_ca - t_apo), R.T) + t_holo

        # 6. 计算新的对齐距离
        distances = torch.norm(apo_ca_aligned - holo_ca, dim=-1)

        # 7. 更新核心掩码
        new_core_mask = (distances < distance_threshold) & valid_mask

        # 如果核心不再发生变化，提前收敛退出
        if torch.equal(new_core_mask, core_mask):
            break

        core_mask = new_core_mask

    return R, t_apo, t_holo, core_mask


def rot_matrix_to_quat(rot_mat):
    """
    将3x3旋转矩阵转化为四元数 (w, x, y, z)
    """
    # 提取矩阵的 9 个元素
    m00, m01, m02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    m10, m11, m12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    m20, m21, m22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]

    # 计算矩阵的迹
    tr = m00 + m11 + m22

    # 初始化四元数张量
    q = torch.zeros(list(rot_mat.shape[:-2]) + [4], device=rot_mat.device, dtype=rot_mat.dtype)

    """=== 分 4 种情况处理，确保数值稳定性 ==="""

    # 情况 1: 迹大于 0 (最常见的情况)
    cond1 = tr > 0
    # 加 clamp 防御极微小的浮点数误差导致负数开根号
    S1 = torch.sqrt(torch.clamp(tr[cond1] + 1.0, min=0.0)) * 2
    q[cond1, 0] = 0.25 * S1
    q[cond1, 1] = (m21[cond1] - m12[cond1]) / S1
    q[cond1, 2] = (m02[cond1] - m20[cond1]) / S1
    q[cond1, 3] = (m10[cond1] - m01[cond1]) / S1

    # 情况 2: m00 最大
    cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
    S2 = torch.sqrt(torch.clamp(1.0 + m00[cond2] - m11[cond2] - m22[cond2], min=0.0)) * 2
    q[cond2, 0] = (m21[cond2] - m12[cond2]) / S2
    q[cond2, 1] = 0.25 * S2
    q[cond2, 2] = (m01[cond2] + m10[cond2]) / S2
    q[cond2, 3] = (m02[cond2] + m20[cond2]) / S2

    # 情况 3: m11 最大
    cond3 = (~cond1) & (~cond2) & (m11 > m22)
    S3 = torch.sqrt(torch.clamp(1.0 + m11[cond3] - m00[cond3] - m22[cond3], min=0.0)) * 2
    q[cond3, 0] = (m02[cond3] - m20[cond3]) / S3
    q[cond3, 1] = (m01[cond3] + m10[cond3]) / S3
    q[cond3, 2] = 0.25 * S3
    q[cond3, 3] = (m12[cond3] + m21[cond3]) / S3

    # 情况 4: m22 最大
    cond4 = (~cond1) & (~cond2) & (~cond3)
    S4 = torch.sqrt(torch.clamp(1.0 + m22[cond4] - m00[cond4] - m11[cond4], min=0.0)) * 2
    q[cond4, 0] = (m10[cond4] - m01[cond4]) / S4
    q[cond4, 1] = (m02[cond4] + m20[cond4]) / S4
    q[cond4, 2] = (m12[cond4] + m21[cond4]) / S4
    q[cond4, 3] = 0.25 * S4

    # 再次 L2 归一化以确保绝对的单位四元数
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    return q


def quat_to_rot_matrix_4d(q):
    """
    将 4D 四元数转化为 3x3 旋转矩阵。
    """
    # 强制 L2 归一化
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]  # 提取分量

    # 预计算平方项与交叉项
    xx, yy, zz = x ** 2, y ** 2, z ** 2
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # 按照标准公式组装矩阵的每一行
    row0 = torch.stack([1.0 - 2 * yy - 2 * zz, 2 * xy - 2 * wz, 2 * xz + 2 * wy], dim=-1)
    row1 = torch.stack([2 * xy + 2 * wz, 1.0 - 2 * xx - 2 * zz, 2 * yz - 2 * wx], dim=-1)
    row2 = torch.stack([2 * xz - 2 * wy, 2 * yz + 2 * wx, 1.0 - 2 * xx - 2 * yy], dim=-1)

    # 形成完整的3x3矩阵
    R = torch.stack([row0, row1, row2], dim=-2)

    return R


def pairwise_distances(x, y=None, return_v=False):
    """
    Args:
        x:  (B, N, d)
        y:  (B, M, d)
    """
    if y is None: y = x
    v = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, d)
    d = safe_norm(v, dim=-1)
    if return_v:
        return d, v
    else:
        return d


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)    # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p


def apply_rotation_to_vector(R, p):
    return local_to_global(R, torch.zeros_like(p), p)


def compose_rotation_and_translation(R1, t1, R2, t2):
    """
    Args:
        R1,t1:  Frame basis and coordinate, (N, L, 3, 3), (N, L, 3).
        R2,t2:  Rotation and translation to be applied to (R1, t1), (N, L, 3, 3), (N, L, 3).
    Returns
        R_new <- R1R2
        t_new <- R1t2 + t1
    """
    R_new = torch.matmul(R1, R2)    # (N, L, 3, 3)
    t_new = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R_new, t_new


def compose_chain(Ts):
    while len(Ts) >= 2:
        R1, t1 = Ts[-2]
        R2, t2 = Ts[-1]
        T_next = compose_rotation_and_translation(R1, t1, R2, t2)
        Ts = Ts[:-2] + [T_next]
    return Ts[0]


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
def quaternion_to_rotation_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
BSD License

For PyTorch3D software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Meta nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
def quaternion_1ijk_to_rotation_matrix(q):
    """
    (1 + ai + bj + ck) -> R
    Args:
        q:  (..., 3)
    """
    b, c, d = torch.unbind(q, dim=-1)
    s = torch.sqrt(1 + b**2 + c**2 + d**2)
    a, b, c, d = 1/s, b/s, c/s, d/s

    o = torch.stack(
        (
            a**2 + b**2 - c**2 - d**2,  2*b*c - 2*a*d,  2*b*d + 2*a*c,
            2*b*c + 2*a*d,  a**2 - b**2 + c**2 - d**2,  2*c*d - 2*a*b,
            2*b*d - 2*a*c,  2*c*d + 2*a*b,  a**2 - b**2 - c**2 + d**2,
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def repr_6d_to_rotation_matrix(x):
    """
    Args:
        x:  6D representations, (..., 6).
    Returns:
        Rotation matrices, (..., 3, 3_index).
    """
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = normalize_vector(a1, dim=-1)
    b2 = normalize_vector(a2 - project_v2v(a2, b1, dim=-1), dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    mat = torch.cat([
        b1.unsqueeze(-1), b2.unsqueeze(-1), b3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999) )
    dihed = torch.nan_to_num(dihed)
    return dihed


def knn_gather(idx, value):
    """
    Args:
        idx:    (B, N, K)
        value:  (B, M, d)
    Returns:
        (B, N, K, d)
    """
    N, d = idx.size(1), value.size(-1)
    idx = idx.unsqueeze(-1).repeat(1, 1, 1, d)      # (B, N, K, d)
    value = value.unsqueeze(1).repeat(1, N, 1, 1)   # (B, N, M, d)
    return torch.gather(value, dim=2, index=idx)


def knn_points(q, p, K):
    """
    Args:
        q: (B, M, d)
        p: (B, N, d)
    Returns:
        (B, M, K), (B, M, K), (B, M, K, d)
    """
    _, L, _ = p.size()
    d = pairwise_distances(q, p)  # (B, N, M)
    dist, idx = d.topk(min(L, K), dim=-1, largest=False)  # (B, M, K), (B, M, K)
    return dist, idx, knn_gather(idx, p)


def angstrom_to_nm(x):
    return x / 10


def nm_to_angstrom(x):
    return x * 10


def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
        chain_nb:   (N, L).
        res_nb:     (N, L).
        mask:       (N, L).
    Returns:
        bb_dihedral:    Omega, Phi, and Psi angles in radian, (N, L, 3).
        mask_bb_dihed:  Masks of dihedral angles, (N, L, 3).
    """
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)  # (N, L)
    omega_mask = torch.logical_not(N_term_flag)
    phi_mask = torch.logical_not(N_term_flag)
    psi_mask = torch.logical_not(C_term_flag)

    # N-termini don't have omega and phi
    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]), 
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0), value=0,
    )

    # C-termini don't have psi
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1), value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed
    return bb_dihedral, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        Inter-residue Phi and Psi angles, (N, L, L, 2).
    """
    N, L = pos_atoms.shape[:2]
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    ir_phi = dihedral_from_four_points(
        pos_C[:,:,None].expand(N, L, L, 3), 
        pos_N[:,None,:].expand(N, L, L, 3), 
        pos_CA[:,None,:].expand(N, L, L, 3), 
        pos_C[:,None,:].expand(N, L, L, 3)
    )
    ir_psi = dihedral_from_four_points(
        pos_N[:,:,None].expand(N, L, L, 3), 
        pos_CA[:,:,None].expand(N, L, L, 3), 
        pos_C[:,:,None].expand(N, L, L, 3), 
        pos_N[:,None,:].expand(N, L, L, 3)
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed


def apply_rotation_matrix_to_rot6d(R, O):
    """
    Args:
        R:  (..., 3, 3)
        O:  (..., 6)
    Returns:
        Rotated 6D representation, (..., 6).
    """
    u1, u2 = O[..., :3, None], O[..., 3:, None] # (..., 3, 1)
    v1 = torch.matmul(R, u1).squeeze(-1)    # (..., 3)
    v2 = torch.matmul(R, u2).squeeze(-1)
    return torch.cat([v1, v2], dim=-1)


def normalize_rot6d(O):
    """
    Args:
        O:  (..., 6)
    """
    u1, u2 = O[..., :3], O[..., 3:]     # (..., 3)
    v1 = F.normalize(u1, p=2, dim=-1)   # (..., 3)
    v2 = F.normalize(u2 - project_v2v(u2, v1), p=2, dim=-1)
    return torch.cat([v1, v2], dim=-1)
    

def reconstruct_backbone(R, t, aa, chain_nb, res_nb, mask):
    """
    Args:
        R:  (N, L, 3, 3)
        t:  (N, L, 3)
        aa: (N, L)
        chain_nb:   (N, L)
        res_nb:     (N, L)
        mask:       (N, L)
    Returns:
        Reconstructed backbone atoms, (N, L, 4, 3).
    """
    N, L = aa.size()
    # atom_coords = restype_heavyatom_rigid_group_positions.clone().to(t) # (21, 14, 3)
    bb_coords = backbone_atom_coordinates_tensor.clone().to(t)  # (21, 3, 3)
    oxygen_coord = bb_oxygen_coordinate_tensor.clone().to(t)    # (21, 3)
    aa = aa.clamp(min=0, max=20)    # 20 for UNK

    bb_coords = bb_coords[aa.flatten()].reshape(N, L, -1, 3)    # (N, L, 3, 3)
    oxygen_coord = oxygen_coord[aa.flatten()].reshape(N, L, -1)  # (N, L, 3)
    bb_pos = local_to_global(R, t, bb_coords)   # Global coordinates of N, CA, C. (N, L, 3, 3).

    # Compute PSI angle
    bb_dihedral, _ = get_backbone_dihedral_angles(bb_pos, chain_nb, res_nb, mask)
    psi = bb_dihedral[..., 2]   # (N, L)
    # Make rotation matrix for PSI
    sin_psi = torch.sin(psi).reshape(N, L, 1, 1)
    cos_psi = torch.cos(psi).reshape(N, L, 1, 1)
    zero = torch.zeros_like(sin_psi)
    one = torch.ones_like(sin_psi)
    row1 = torch.cat([one, zero, zero], dim=-1)     # (N, L, 1, 3)
    row2 = torch.cat([zero, cos_psi, -sin_psi], dim=-1) # (N, L, 1, 3)
    row3 = torch.cat([zero, sin_psi, cos_psi], dim=-1)  # (N, L, 1, 3)
    R_psi = torch.cat([row1, row2, row3], dim=-2)       # (N, L, 3, 3)

    # Compute rotoation and translation of PSI frame, and position of O.
    R_psi, t_psi = compose_chain([
        (R, t), # Backbone
        (R_psi, torch.zeros_like(t)),       # PSI angle
    ])
    O_pos = local_to_global(R_psi, t_psi, oxygen_coord.reshape(N, L, 1, 3))

    bb_pos = torch.cat([bb_pos, O_pos], dim=2)  # (N, L, 4, 3)
    return bb_pos
    

def reconstruct_backbone_partially(pos_ctx, R_new, t_new, aa, chain_nb, res_nb, mask_atoms, mask_recons):
    """
    Args:
        pos:    (N, L, A, 3).
        R_new:  (N, L, 3, 3).
        t_new:  (N, L, 3).
        mask_atoms: (N, L, A).
        mask_recons:(N, L).
    Returns:
        pos_new:    (N, L, A, 3).
        mask_new:   (N, L, A).
    """
    N, L, A = mask_atoms.size()

    mask_res = mask_atoms[:, :, BBHeavyAtom.CA]
    pos_recons = reconstruct_backbone(R_new, t_new, aa, chain_nb, res_nb, mask_res) # (N, L, 4, 3)
    pos_recons = F.pad(pos_recons, pad=(0, 0, 0, A-4), value=0) # (N, L, A, 3)

    pos_new = torch.where(
        mask_recons[:, :, None, None].expand_as(pos_ctx),
        pos_recons, pos_ctx
    )   # (N, L, A, 3)

    mask_bb_atoms = torch.zeros_like(mask_atoms)
    mask_bb_atoms[:, :, :4] = True
    mask_new = torch.where(
        mask_recons[:, :, None].expand_as(mask_atoms),
        mask_bb_atoms, mask_atoms
    )

    return pos_new, mask_new


def center_and_realign_missing_complex(xyz, mask_t, chain_id):
    """
    xyz:     (L, n_atom, 3)
    mask_t:  (L, n_atom)
    chain_id:(L,)，每个残基所属链/组分的ID（相同ID表示同一链）
    """
    L = xyz.shape[0]
    device = xyz.device
    chain_id = chain_id.to(device)

    mask = mask_t[..., 1]  # (L,) 只要 CA 原子有效，该 token 就参与全局居中

    # 全局居中（跨链一起平移，包含有效蛋白质和有效配体）
    denom = mask.float().sum().clamp_min(1.0)
    center_CA = (xyz[:, 1] * mask[:, None].float()).sum(dim=0) / denom  # (3,)
    xyz = torch.where(mask.view(L, 1, 1), xyz - center_CA.view(1, 1, 3), xyz)

    # 把缺失残基/配体移动到“本链最近的有效残基”的CA附近
    for cid in torch.unique(chain_id):
        chain_sel = (chain_id == cid)  # (L,)
        valid_sel = chain_sel & mask  # 本链有效节点
        missing_sel = chain_sel & (~mask)  # 本链缺失节点

        if not torch.any(missing_sel):
            continue

        pos = torch.where(chain_sel)[0]  # 本链所有残基/配体索引

        if not torch.any(valid_sel):
            # 若整条链完全缺失，此时应将完全缺失的链放置在当前坐标系的原点(全局中心)，这样保证它不会被遗留在平移前的旧坐标空间中。
            xyz[pos] = torch.where(mask[pos].view(-1, 1, 1), xyz[pos], torch.zeros_like(xyz[pos]))
            continue

        exist = torch.where(valid_sel)[0]  # (Lsub,)

        # 在“本链残基集合 pos”内，为每个位置找最近 exist（按序号距离）
        dist = (pos[:, None] - exist[None, :]).abs()  # (Lc, Lsub)
        nearest = exist[dist.argmin(dim=-1)]  # (Lc,)

        offset_CA = xyz[nearest, 1]  # (Lc, 3)

        # 将缺失原子的坐标直接覆盖为 offset_CA。
        xyz[pos] = torch.where(mask[pos].view(-1, 1, 1), xyz[pos], offset_CA.view(-1, 1, 3))

    return xyz, center_CA
