import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import model.utils.util_module as util_module
import model.utils.chemical as chemical
from icecream import ic
from opt_einsum import contract as einsum
from contextlib import ExitStack
from .utils.utils import xyz_frame_from_rotation_mask, get_time_embedding
from .layers.Attention_module import *
from .layers.SE3_network import SE3TransformerWrapper
from .loss.loss import (
    calc_lj_grads, calc_chiral_grads
)


# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_pair, minpos=-32, maxpos=32, maxpos_atom=8, p_drop=0.15, use_same_chain=False, enable_same_chain=False):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.maxpos_atom = maxpos_atom
        self.nbin_res = abs(minpos)+maxpos+2 # include 0 and "unknown" value (maxpos+1)
        self.nbin_atom = maxpos_atom+2 # include 0 and "unknown" token (maxpos_sm + 1)
        self.emb_res = nn.Embedding(self.nbin_res, d_pair)
        self.emb_atom = nn.Embedding(self.nbin_atom, d_pair)

        # 用于将噪声化的键特征投影到 pair 维度
        self.bond_emb = nn.Linear(chemical.NBTYPES, d_pair)

        self.use_same_chain = use_same_chain
        if use_same_chain:
            self.emb_chain = nn.Embedding(2, d_pair)
        self.enable_same_chain = enable_same_chain

    def forward(self, idx, bond_feats, dist_matrix, sm_mask, is_protein, is_atomize_protein, same_chain=None):
        """
        构建二维距离特征
        :param idx:
        :param bond_feats:
        :param dist_matrix:
        :param sm_mask: 标记配体部分，排除共价结合中原子化残基的原子
        :param is_protein:  标记配体部分，包含了共价结合中原子化残基的原子
        :param same_chain:
        :return:
        """
        res_dist, atom_dist = util_module.get_res_atom_dist(idx, bond_feats, dist_matrix, ~is_protein[0], is_atomize_protein[0],
            minpos_res=self.minpos, maxpos_res=self.maxpos, maxpos_atom=self.maxpos_atom)

        # 氨基酸距离分桶
        bins = torch.arange(self.minpos, self.maxpos+1, device=bond_feats.device)
        ib_res = torch.bucketize(res_dist, bins).long()  # (B, L, L)
        emb_res = self.emb_res(ib_res)  # (B, L, L, d_pair)

        # 配体距离分桶
        bins = torch.arange(0, self.maxpos_atom+1, device=bond_feats.device)
        ib_atom = torch.bucketize(atom_dist, bins).long()  # (B, L, L)
        emb_atom = self.emb_atom(ib_atom)  # (B, L, L, d_pair)

        # 构建 2D mask: (B, L, L)，如果是配体-配体对，则为 True
        sm_mask = torch.where(is_atomize_protein, True, sm_mask)  # 为保证尺度统一，原子化残基的距离桶特征也进行屏蔽
        ligand_pair_mask = sm_mask[:, :, None] & sm_mask[:, None, :]
        emb_atom = emb_atom * (~ligand_pair_mask).unsqueeze(-1).float()  # 将配体部分的嵌入置0

        bond_matrix = nn.functional.one_hot(bond_feats, num_classes=chemical.NBTYPES).float()
        out = emb_res + emb_atom + self.bond_emb(bond_matrix)

        if self.use_same_chain and self.enable_same_chain == False and same_chain is not None:
            emb_c = self.emb_chain(same_chain.long())
            out += emb_c*0 # cursed but exists for backwards compatibility
        elif self.enable_same_chain == True:
            emb_c = self.emb_chain(same_chain.long())
            out += emb_c

        return out


# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16, d_rbf=64,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.emb_rbf = nn.Linear(d_rbf, d_pair)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = util_module.Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.emb_rbf= util_module.init_lecun_normal(self.emb_rbf)
        self.proj_state = util_module.init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.emb_rbf.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, 36)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = pair + self.emb_rbf(rbf_feat)
        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.type_as(state)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa


def find_symmsub(Ltot, Lasu, k):
    """
    Creates a symmsub matrix 
    
    Parameters:
        Ltot (int, required): Total length of all residues 
        
        Lasu (int, required): Length of asymmetric units
        
        k (int, required): Number of off diagonals to include in symmetrization
    
    
    """
    assert Ltot % Lasu == 0 
    nchunk = Ltot // Lasu 

    N = 2*k + 1 # total number of diagonals being accessed 
    symmsub = torch.ones((nchunk, nchunk))*-1
    C = 0 # a marker for blocks of the same category 

    for i in range(N):                                # i      = 0, 1,2, 3,4, 5,6...
        offset = int(((i+1) // 2) * (math.pow(-1,i))) # offset = 0,-1,1,-2,2,-3,3...

        row = torch.arange(nchunk)
        col = torch.roll(row, offset)

        if offset < 0:
            row = row[:-abs(offset)]
            col = col[:-abs(offset)]
        elif offset > 0:
            row = row[abs(offset):]
            col = col[abs(offset):]
        else:# i=0
            pass 

        symmsub[row, col] = i

    return symmsub.long()


def max_block_activations(pair, symmsub):
    """
    copies pair activations around in blocks according to 
    matrix S
    """
    B,L = pair.shape[:2]

    Osub = symmsub.shape[0]

    # average pairs/blocks together 
    Leff = L//Osub

    # applies block averaging to the pair representation based on symmsub
    # pairsymm = torch.zeros([Osub,Leff,Leff,pair.shape[-1]], device=pair.device, dtype=pair.dtype)
    # Nsymm    = torch.zeros([Osub], device=pair.device, dtype=torch.int)

    stacks = {}

    # find all of the activation blocks
    for i in range(Osub):
        for j in range(Osub):
            sij = symmsub[i,j]
            if (sij>=0):
                if not stacks.get(int(sij), False):
                    stacks[int(sij)] = []
                stacks[int(sij)].append( pair[0, i*Leff:(i+1)*Leff, j*Leff:(j+1)*Leff] )

    # make tensors and find max activation in each tensor 
    # ic(list(stacks.keys()))
    for key,val in stacks.items():
        A = torch.stack(stacks[key]) # stacked block activations 
        B,max_idx = torch.max(A, dim=0)      # find the max 
        stacks[key] = B              # replace with the max 

    for i in range(Osub):
            for j in range(Osub):
                sij = symmsub[i,j]
                if (sij>=0):
                    pair[0, i*Leff:(i+1)*Leff, j*Leff:(j+1)*Leff] = stacks[int(sij)] #pairsymm[sij]/Nsymm[sij]
    return pair 


def mean_block_activations(pair, symmsub):
    """
    Applies block average symmetrization 
    """
    B,L = pair.shape[:2]

    Osub = symmsub.shape[0]

    # average pairs/blocks together 
    Leff = L//Osub

    # applies block averaging to the pair representation based on symmsub
    pairsymm = torch.zeros([Osub,Leff,Leff,pair.shape[-1]], device=pair.device, dtype=pair.dtype)
    Nsymm = torch.zeros([Osub], device=pair.device, dtype=torch.int)

    for i in range(Osub):
        for j in range(Osub):
            sij = symmsub[i,j]
            if (sij>=0):
                pairsymm[sij] += pair[0, i*Leff:(i+1)*Leff, j*Leff:(j+1)*Leff]
                Nsymm[sij]    += 1

    for i in range(Osub):
        for j in range(Osub):
            sij = symmsub[i,j]
            if (sij>=0):
                pair[0, i*Leff:(i+1)*Leff, j*Leff:(j+1)*Leff] = pairsymm[sij]/Nsymm[sij]

    return pair


def apply_pair_symmetry(pair, symmsub, method='mean', main_block=None):
    """
    Applies pair symmetrizing operation
    """
    assert method in ['mean','max','copy']

    if method == 'mean': 
        pair = mean_block_activations(pair, symmsub) 

    elif method == 'copy':
        assert not (main_block is None), "cant have None main block here" 
        pair = copy_block_activations(pair, symmsub, main_block=main_block)

    elif method == 'max':
        pair = max_block_activations(pair, symmsub)

    return pair


class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_hidden_state=16, d_rbf=64, d_state=32, p_drop=0.15,
                 symmetrize_repeats=False, repeat_length=None, symmsub_k=1, sym_method='max',main_block=None):
        """
        
        Parameters:
            symmetrize_repeats (bool, optional): whether to symmetrize the repeats. 

            repeat_length (int, optional): length of the repeat unit in repeat protein 

            symmsub_k (int, optional): number of diagonals to use for symmetrization

            sym_method (str, optional): method to use for symmetrization.

            main_block (int, optional): main block to use for symmetrization (the one with the motif)
        """
        super(PairStr2Pair, self).__init__()

        self.symmetrize_repeats = symmetrize_repeats
        self.repeat_length = repeat_length
        self.symmsub_k = symmsub_k
        self.sym_method = sym_method
        self.main_block = main_block

        self.norm_state = nn.LayerNorm(d_state)
        self.proj_left = nn.Linear(d_state, d_hidden_state)
        self.proj_right = nn.Linear(d_state, d_hidden_state)
        self.to_gate = nn.Linear(d_hidden_state*d_hidden_state, d_pair)

        self.emb_rbf = nn.Linear(d_rbf, d_pair)

        self.drop_row = util_module.Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = util_module.Dropout(broadcast_dim=2, p_drop=p_drop)

        self.tri_mul_out = TriangleMultiplication(d_pair, d_hidden=d_hidden)
        self.tri_mul_in = TriangleMultiplication(d_pair, d_hidden, outgoing=False)

        self.row_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_pair, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)

        self.reset_parameter()

    def reset_parameter(self):
        self.emb_rbf = util_module.init_lecun_normal(self.emb_rbf)
        nn.init.zeros_(self.emb_rbf.bias)

        self.proj_left = util_module.init_lecun_normal(self.proj_left)
        nn.init.zeros_(self.proj_left.bias)
        self.proj_right = util_module.init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_right.bias)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.ones_(self.to_gate.bias)

    # perform a striped p2p op
    def subblock(self, OP, pair, rbf_feat, crop):
        N,L = pair.shape[:2]

        nbox = (L-1)//(crop//2)+1
        idx = torch.triu_indices(nbox,nbox,1, device=pair.device)
        ncrops = idx.shape[1]

        pairnew = torch.zeros((N,L*L,pair.shape[-1]), device=pair.device, dtype=pair.dtype)
        countnew = torch.zeros((N,L*L), device=pair.device, dtype=torch.int)

        for i in range(ncrops):
            # reindex sub-blocks
            offsetC = torch.clamp( (1+idx[1,i:(i+1)])*(crop//2)-L, min=0 ) # account for going past L
            offsetN = torch.zeros_like(offsetC)
            mask = (offsetC>0)*((idx[0,i]+1)==idx[1,i])
            offsetN[mask] = offsetC[mask]
            pairIdx = torch.zeros((1,crop), dtype=torch.long, device=pair.device)
            pairIdx[:,:(crop//2)] = torch.arange(crop//2, dtype=torch.long, device=pair.device)+idx[0,i:(i+1),None]*(crop//2) - offsetN[:,None]
            pairIdx[:,(crop//2):] = torch.arange(crop//2, dtype=torch.long, device=pair.device)+idx[1,i:(i+1),None]*(crop//2) - offsetC[:,None]

            # do reindexing
            iL,iU = pairIdx[:,:,None], pairIdx[:,None,:]
            paircrop = pair[:,iL,iU,:].reshape(-1,crop,crop,pair.shape[-1])
            rbfcrop = rbf_feat[:,iL,iU,:].reshape(-1,crop,crop,rbf_feat.shape[-1])

            # row attn
            paircrop = OP(paircrop, rbfcrop).to(pair.dtype)

            # unindex
            iUL = (iL*L+iU).flatten()
            pairnew.index_add_(1,iUL, paircrop.reshape(N,iUL.shape[0],pair.shape[-1]))
            countnew.index_add_(1,iUL, torch.ones((N,iUL.shape[0]), device=pair.device, dtype=torch.int))

        return pair + (pairnew/countnew[...,None]).reshape(N,L,L,-1)

    def forward(self, pair, rbf_feat, state, crop=-1):
        B,L = pair.shape[:2]

        rbf_feat = self.emb_rbf(rbf_feat) # B, L, L, d_pair

        state = self.norm_state(state)
        left = self.proj_left(state)
        right = self.proj_right(state)
        gate = einsum('bli,bmj->blmij', left, right).reshape(B,L,L,-1)
        gate = torch.sigmoid(self.to_gate(gate))
        rbf_feat = gate*rbf_feat


        crop = 2*(crop//2) # make sure even

        if (crop>0 and crop<=L):
            pair = self.subblock( 
                lambda x,y:self.drop_row(self.tri_mul_out(x)),
                pair, rbf_feat, crop
            )

            pair = self.subblock( 
                lambda x,y:self.drop_row(self.tri_mul_in(x)), 
                pair, rbf_feat, crop
            )

            pair = self.subblock( 
                lambda x,y:self.drop_row(self.row_attn(x,y)), 
                pair, rbf_feat, crop
            )

            pair = self.subblock( 
                lambda x,y:self.drop_col(self.col_attn(x,y)), 
                pair, rbf_feat, crop
            )

            # feed forward layer
            RESSTRIDE = 16384//L
            for i in range((L-1)//RESSTRIDE+1):
                r_i,r_j = i*RESSTRIDE, min((i+1)*RESSTRIDE,L)
                pair[:,r_i:r_j] = pair[:,r_i:r_j] + self.ff(pair[:,r_i:r_j])

        else:
            #_nc = lambda x:torch.sum(torch.isnan(x))
            pair = pair + self.drop_row(self.tri_mul_out(pair)) 
            pair = pair + self.drop_row(self.tri_mul_in(pair)) 
            pair = pair + self.drop_row(self.row_attn(pair, rbf_feat)) 
            pair = pair + self.drop_col(self.col_attn(pair, rbf_feat)) 
            pair = pair + self.ff(pair)
        
        # symmetry/repeat proteins (Diffusion inference only)
        if self.symmetrize_repeats:
            symmsub = find_symmsub(L, self.repeat_length, self.symmsub_k)
        else:
            symmsub = None

        if symmsub is not None:
            pair = apply_pair_symmetry(pair, symmsub, self.sym_method, self.main_block)
        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=16, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = util_module.init_lecun_normal(self.proj_left)
        self.proj_right = util_module.init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
        
        pair = pair + out

        return pair

class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, d_rbf=64,
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, 
            nextra_l0=0, nextra_l1=0, p_drop=0.1
    ):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)

        self.timestep_film = nn.Linear(SE3_param['l0_in_features'], SE3_param['l0_in_features'] * 2)

        self.embed_node = nn.Linear(d_msa + d_state, SE3_param['l0_in_features'])
        self.ff_node = FeedForwardLayer(SE3_param['l0_in_features'], 2, p_drop=p_drop)
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])

        self.embed_edge = nn.Linear(d_pair+d_rbf+1, SE3_param['num_edge_features'])
        self.ff_edge = FeedForwardLayer(SE3_param['num_edge_features'], 2, p_drop=p_drop)
        self.norm_edge = nn.LayerNorm(SE3_param['num_edge_features'])

        SE3_param_temp = SE3_param.copy()
        SE3_param_temp['l0_in_features'] += nextra_l0
        SE3_param_temp['l1_in_features'] += nextra_l1
        
        self.se3 = SE3TransformerWrapper(**SE3_param_temp)

        self.sc_predictor = SCPred(
            d_msa=d_msa,
            d_state=SE3_param['l0_out_features'],
            p_drop=p_drop)

        self.nextra_l0 = nextra_l0
        self.nextra_l1 = nextra_l1

        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_node = util_module.init_lecun_normal(self.embed_node)
        self.embed_edge = util_module.init_lecun_normal(self.embed_edge)
        self.timestep_film = util_module.init_lecun_normal(self.timestep_film)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_node.bias)
        nn.init.zeros_(self.embed_edge.bias)
        nn.init.zeros_(self.timestep_film.bias)
    
    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, t_emb, msa, pair, xyz, state, idx, rotation_mask, bond_feats, dist_matrix, atom_frames, is_atomize_protein,
                extra_l0=None, extra_l1=None, use_atom_frames=False, top_k=128, use_checkpoint=False):
        # process msa & pair features
        B, N, L = msa.shape[:3]
        seq = self.norm_msa(msa[:,0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)

        # 初始化节点特征
        node = torch.cat((seq, state), dim=-1)
        node = self.embed_node(node)
        node = node + self.ff_node(node)
        node = self.norm_node(node)

        # 使用FiLM将时间步嵌入集成到节点特征中
        time_c = self.timestep_film(t_emb)
        scale, shift = time_c.chunk(2, dim=-1)
        node = torch.add(node * (scale + 1.), shift)

        # 构建 2D mask: (B, L, L)，如果是配体-配体对，则为 True
        ligand_pair_mask = rotation_mask[:, :, None] & rotation_mask[:, None, :]

        neighbor = util_module.get_seqsep_protein_sm(idx, bond_feats, dist_matrix, rotation_mask, is_atomize_protein[0])
        neighbor = neighbor * (~ligand_pair_mask).unsqueeze(-1).float()  # 将配体部分的邻居关系mask掉
        cas = xyz[:, :, 1].contiguous()
        rbf_feat = util_module.rbf(torch.cdist(cas, cas))

        # 初始化边特征
        edge = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        edge = self.embed_edge(edge)
        edge = edge + self.ff_edge(edge)
        edge = self.norm_edge(edge)
        
        # 初始化图
        if top_k > 0:
            G, edge_feats = util_module.make_topk_graph(xyz[:,:,1,:], edge, idx, top_k=top_k)
        else:
            G, edge_feats = util_module.make_full_graph(xyz[:,:,1,:], edge, idx)

        if use_atom_frames:  # 配体l1特征为指向邻近原子的向量
            xyz_frame = xyz_frame_from_rotation_mask(xyz, rotation_mask, atom_frames)
            l1_feats = xyz_frame - xyz_frame[:, :, 1, :].unsqueeze(2)
        else:
            l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
            # 配体部分的l1_feats置0，防止将配体部分强行伪装成相对方向特征
            l1_feats = l1_feats * (~rotation_mask.unsqueeze(-1).unsqueeze(-1)).float()
        l1_feats = l1_feats.reshape(B*L, -1, 3)

        if extra_l1 is not None:
            l1_feats = torch.cat((l1_feats, extra_l1), dim=1)
        if extra_l0 is not None:
            node = torch.cat((node, extra_l0), dim=2)

        # 使用SE(3)transformer并更新坐标
        if use_checkpoint:
            shift = checkpoint.checkpoint(util_module.create_custom_forward(self.se3), G, node.reshape(B * L, -1, 1),
                                          l1_feats, edge_feats, use_reentrant=True)
        else:
            shift = self.se3(G, node.reshape(B * L, -1, 1), l1_feats, edge_feats)

        state = shift['0'].reshape(B, L, -1)  # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3)
        # offset[:, is_motif, ...] = 0            # NOTE: DJ - frozen motif!! 保持motif部分不变

        T = offset[:,:,0,:] / 10.  # 平移向量
        R = offset[:,:,1,:] / 100.  # 旋转矩阵

        Qnorm = torch.sqrt(1 + torch.sum(R * R, dim=-1))
        qA, qB, qC, qD = 1 / Qnorm, R[:, :, 0] / Qnorm, R[:, :, 1] / Qnorm, R[:, :, 2] / Qnorm

        # 使用四元数分量更新旋转矩阵
        v = xyz - xyz[:, :, 1:2, :]
        Rout = torch.zeros((B, L, 3, 3), device=xyz.device)
        Rout[:, :, 0, 0] = qA * qA + qB * qB - qC * qC - qD * qD
        Rout[:, :, 0, 1] = 2 * qB * qC - 2 * qA * qD
        Rout[:, :, 0, 2] = 2 * qB * qD + 2 * qA * qC
        Rout[:, :, 1, 0] = 2 * qB * qC + 2 * qA * qD
        Rout[:, :, 1, 1] = qA * qA - qB * qB + qC * qC - qD * qD
        Rout[:, :, 1, 2] = 2 * qC * qD - 2 * qA * qB
        Rout[:, :, 2, 0] = 2 * qB * qD - 2 * qA * qC
        Rout[:, :, 2, 1] = 2 * qC * qD + 2 * qA * qB
        Rout[:, :, 2, 2] = qA * qA - qB * qB - qC * qC + qD * qD
        I = torch.eye(3, device=Rout.device).expand(B, L, 3, 3)
        Rout = torch.where(torch.where(is_atomize_protein, True, rotation_mask).reshape(B, L, 1, 1), I, Rout)  # 配体部分（包括原子化残基）不参与旋转
        xyz = torch.einsum('blij,blaj->blai', Rout, v) + xyz[:, :, 1:2, :] + T[:, :, None, :]

        alpha = self.sc_predictor(msa[:,0], state)
        # quat = torch.stack([qA, qB, qC, qD], dim=2)  # 拼接为四元数
        return xyz, state, alpha


class AllatomEmbed(nn.Module):
    def __init__(
        self,
        d_state_in=64, 
        d_state_out=32,
        p_mask=0.15
    ):
        super(AllatomEmbed, self).__init__()

        self.p_mask = p_mask

        # initial node & pair feature process
        self.compress_embed = nn.Linear(d_state_in + 29, d_state_out) # 29->5 if using element
        self.norm_state = nn.LayerNorm(d_state_out)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.compress_embed = util_module.init_lecun_normal(self.compress_embed)
        # initialize bias to zeros
        nn.init.zeros_(self.compress_embed.bias)

    def forward(self, state, seq, eltmap):
        B,L = state.shape[:2]
        mask = torch.rand(B,L) < self.p_mask
        state = state.reshape(B,L,1,-1).repeat(1,1,27,1)
        state[mask] = 0.0
        elements = F.one_hot(eltmap[seq], num_classes=29)  # 29->5 if using element
        state = self.compress_embed(
            torch.cat( (state,elements), dim=-1 )
        )
        state = self.norm_state( state )

        return state

# embed residue state + atomtype -> per-atom state
# 
class AllatomEmbed(nn.Module):
    def __init__(
        self,
        d_state_in=64, 
        d_state_out=32,
        p_mask=0.15
    ):
        super(AllatomEmbed, self).__init__()

        self.p_mask = p_mask

        # initial node & pair feature process
        self.compress_embed = nn.Linear(d_state_in + 29, d_state_out) # 29->5 if using element
        self.norm_state = nn.LayerNorm(d_state_out)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.compress_embed = util_module.init_lecun_normal(self.compress_embed)
        # initialize bias to zeros
        nn.init.zeros_(self.compress_embed.bias)

    def forward(self, state, seq, eltmap):
        B,L = state.shape[:2]
        mask = torch.rand(B,L) < self.p_mask
        state = state.reshape(B,L,1,-1).repeat(1,1,27,1)
        state[mask] = 0.0
        elements = F.one_hot(eltmap[seq], num_classes=29)  # 29->5 if using element
        state = self.compress_embed(
            torch.cat( (state,elements), dim=-1 )
        )
        state = self.norm_state( state )

        return state

# embed per-atom state -> residue state
class ResidueEmbed(nn.Module):
    def __init__(
        self,
        d_state_in=16,
        d_state_out=64
    ):
        super(ResidueEmbed, self).__init__()

        self.compress_embed = nn.Linear(27*d_state_in, d_state_out)
        self.norm_state = nn.LayerNorm(d_state_out)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.compress_embed = util_module.init_lecun_normal(self.compress_embed)
        # initialize bias to zeros
        nn.init.zeros_(self.compress_embed.bias)

    def forward(self, state):
        B,L = state.shape[:2]

        state = self.compress_embed( state.reshape(B,L,-1) )
        state = self.norm_state( state )

        return state

class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.linear_out = nn.Linear(d_hidden, 2 * chemical.NTOTALDOFS)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.linear_s0 = util_module.init_lecun_normal(self.linear_s0)
        self.linear_si = util_module.init_lecun_normal(self.linear_si)
        self.linear_out = util_module.init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)
        
        # right before relu activation: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # right before residual connection: zero initialize
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq, state):
        '''
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: hidden embeddings corresponding to query sequence (B, L, d_msa)
            - state: state feature (output l0 feature) from previous SE3 layer (B, L, d_state)
        Outputs:
            - si: predicted torsion/pseudotorsion angles (phi, psi, omega, chi1~4 with cos/sin, theta) (B, L, NTOTALDOFS, 2)
        '''
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        si = self.linear_out(F.relu_(si))
        return si.view(B, L, chemical.NTOTALDOFS, 2)

def normQ(Q):
    """normalize a quaternions
    """
    return Q / torch.linalg.norm(Q, keepdim=True, dim=-1)

def Qs2Rs(Qs):
    Rs = torch.zeros((*Qs.shape[:-1],3,3), device=Qs.device)

    Rs[...,0,0] = Qs[...,0]*Qs[...,0]+Qs[...,1]*Qs[...,1]-Qs[...,2]*Qs[...,2]-Qs[...,3]*Qs[...,3]
    Rs[...,0,1] = 2*Qs[...,1]*Qs[...,2] - 2*Qs[...,0]*Qs[...,3]
    Rs[...,0,2] = 2*Qs[...,1]*Qs[...,3] + 2*Qs[...,0]*Qs[...,2]
    Rs[...,1,0] = 2*Qs[...,1]*Qs[...,2] + 2*Qs[...,0]*Qs[...,3]
    Rs[...,1,1] = Qs[...,0]*Qs[...,0]-Qs[...,1]*Qs[...,1]+Qs[...,2]*Qs[...,2]-Qs[...,3]*Qs[...,3]
    Rs[...,1,2] = 2*Qs[...,2]*Qs[...,3] - 2*Qs[...,0]*Qs[...,1]
    Rs[...,2,0] = 2*Qs[...,1]*Qs[...,3] - 2*Qs[...,0]*Qs[...,2]
    Rs[...,2,1] = 2*Qs[...,2]*Qs[...,3] + 2*Qs[...,0]*Qs[...,1]
    Rs[...,2,2] = Qs[...,0]*Qs[...,0]-Qs[...,1]*Qs[...,1]-Qs[...,2]*Qs[...,2]+Qs[...,3]*Qs[...,3]

    return Rs

"""该函数假设是完美对称的，因此目前在本项目不推荐在蛋白质构象变化预测中启用该函数，但保留该函数留做后续讨论"""
def update_symm_Rs(xyz, Lasu, symmsub, symmRs, fit=False, tscale=1.0):
    """
    将模型输出的单链坐标扩展为完整的多聚体结构
    xyz：[B, L, n_atom, 3]
    Lasu：不对称单元长度。函数使用xyz[:, :Lasu]来截取第一个单体作为模板，进而生成其它对称副本
    symmsub：一个索引张量，用于从 symmRs 中选择当前需要的对称操作矩阵。
    symmRs：[N_symm_ops, 3, 3]，对称操作矩阵库。它存储了特定对称群的所有旋转矩阵。
    fit：是否开启对称性拟合。如果为 True，函数将使用 LBFGS 优化器来微调 ASU 的全局旋转和平移，以确保生成的完整结构在物理和几何上更符合对称约束。
    tscale在优化过程中，用于缩放平移向量T_0的步长或权重。
    """
    def dist_error_comp(R0,T0,xyz,tscale):
        B = xyz.shape[0]
        Tcom = xyz[:,:Lasu].mean(dim=1,keepdim=True)
        Tcorr = torch.einsum('ij,brj->bri', R0, xyz[:,:Lasu]-Tcom) + Tcom + tscale*T0[None,None,:]

        # distance map loss
        Xsymm = torch.einsum('sij,brj->bsri', symmRs[symmsub], Tcorr).reshape(B,-1,3)
        # 由于没有找到Ts的定义，根据名字来看，可能是平移向量（CA原子的坐标）
        # Xtrue = Ts
        Xtrue = xyz[:, :, 1]

        delsx = Xsymm[:,:Lasu,None]-Xsymm[:, None, Lasu:]
        deltx = Xtrue[:,:Lasu,None]-Xtrue[:, None, Lasu:]
        dsymm = torch.linalg.norm(delsx, dim=-1)
        dtrue = torch.linalg.norm(deltx, dim=-1)
        loss1 = torch.abs(dsymm-dtrue).mean()

        return loss1,0.0 #loss2

    def dist_error(R0,T0,xyz,tscale,w_clash=0.0):
        l1,l2 = dist_error_comp(R0,T0,xyz,tscale)
        return l1+w_clash*l2

    def Q2R(Q):
        Qs = torch.cat((torch.ones((1),device=Q.device),Q),dim=-1)
        Qs = normQ(Qs)
        return Qs2Rs(Qs[None,:]).squeeze(0)
        

    B = xyz.shape[0]

    # symmetry correction 1: don't let COM (of entire complex) move
    #Tmean = xyz[:,:Lasu].reshape(-1,3).mean(dim=0)
    #Tmean = torch.einsum('sij,j->si', symmRs, Tmean).mean(dim=0)
    #xyz = xyz - Tmean

    if fit:
        # symmetry correction 2: use minimization to minimize drms
        with torch.enable_grad():
            T0 = torch.zeros(3,device=xyz.device).requires_grad_(True)
            Q0 = torch.zeros(3,device=xyz.device).requires_grad_(True)
            lbfgs = torch.optim.LBFGS([T0,Q0],
                        history_size=10, 
                        max_iter=4,
                        line_search_fn="strong_wolfe")
            def closure():
                lbfgs.zero_grad()
                loss = dist_error(Q2R(Q0),T0,xyz[:,:,1], tscale)
                loss.backward() #retain_graph=True)
                return loss

            for e in range(4):
                loss = lbfgs.step(closure)

            Tcom = xyz[:,:Lasu].mean(dim=1,keepdim=True).detach()
            Q0 = Q0.detach()
            T0 = T0.detach()
            xyz = torch.einsum('ij,braj->brai', Q2R(Q0), xyz[:,:Lasu]-Tcom) +Tcom + tscale*T0[None,None,:]

    xyz = torch.einsum('sij,braj->bsrai', symmRs[symmsub], xyz[:,:Lasu])
    xyz = xyz.reshape(B,-1,3,3) # (B,S,L,3,3)

    return xyz

def update_symm_subs(xyz, pair, symmids, symmsub, symmRs, metasymm, fit=False, tscale=1.0):
    """
    与 update_symm_Rs 配合使用，主要作用是动态更新对称子单元的索引映射关系
    """
    B,Ls = xyz.shape[0:2]
    Osub = symmsub.shape[0]
    L = Ls//Osub

    com = xyz[:,:L,1].mean(dim=-2)
    rcoms = torch.einsum('sij,bj->si', symmRs, com)
    subsymms, nneighs = metasymm
    symmsub_new = []
    for i in range(len(subsymms)):
        drcoms = torch.linalg.norm(rcoms[0,:] - rcoms[subsymms[i],:], dim=-1)
        _,subs_i = torch.topk(drcoms,nneighs[i],largest=False)
        subs_i,_ = torch.sort( subsymms[i][subs_i] )
        symmsub_new.append(subs_i)

    symmsub_new = torch.cat(symmsub_new)

    s_old = symmids[symmsub[:,None],symmsub[None,:]]
    s_new = symmids[symmsub_new[:,None],symmsub_new[None,:]]

    # remap old->new
    # a) find highest-magnitude patches
    pairsub = dict()
    pairmag = dict()
    for i in range(Osub):
        for j in range(Osub):
            idx_old = s_old[i,j].item()
            sub_ij = pair[:,i*L:(i+1)*L,j*L:(j+1)*L,:].clone()
            mag_ij = torch.max(sub_ij.flatten()) #torch.norm(sub_ij.flatten())
            if idx_old not in pairsub or mag_ij > pairmag[idx_old]:
                pairmag[idx_old] = mag_ij
                pairsub[idx_old] = (i,j) #sub_ij

    # b) reindex
    idx = (
        torch.arange(Osub*L,device=pair.device)[:,None]*Osub*L
         + torch.arange(Osub*L,device=pair.device)[None,:]
    )
    for i in range(Osub):
        for j in range(Osub):
            idx_new = s_new[i,j].item()
            if idx_new in pairsub:
                inew,jnew = pairsub[idx_new]
                idx[i*L:(i+1)*L,j*L:(j+1)*L] = (
                    Osub*L*torch.arange(inew*L,(inew+1)*L)[:,None]
                    + torch.arange(jnew*L,(jnew+1)*L)[None,:]
                )
    pair = pair.view(1,-1,pair.shape[-1])[:,idx.flatten(),:].view(1,Osub*L,Osub*L,pair.shape[-1])

    # if symmsub is not None and symmsub.shape[0]>1:
    #     xyz = update_symm_Rs(xyz, L, symmsub_new, symmRs, fit, tscale)

    return xyz, pair, symmsub_new


class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_rbf=64,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, 
                 minpos=-32, maxpos=32, maxpos_atom=8, p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 nextra_l0=0, nextra_l1=0, use_same_chain=False, enable_same_chain=False,
                 symmetrize_repeats=None, repeat_length=None,symmsub_k=None, sym_method=None, main_block=None,
                 fit=False, tscale=1.0
                 ):

        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.fit = fit
        self.tscale = tscale

        self.pos = PositionalEncoding2D(d_rbf, minpos=minpos, maxpos=maxpos, 
                                        maxpos_atom=maxpos_atom, p_drop=p_drop, 
                                        use_same_chain=use_same_chain,
                                        enable_same_chain=enable_same_chain)

        self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair, d_rbf=d_rbf,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)

        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                 d_hidden=d_hidden//2, p_drop=p_drop)   

        self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair, d_rbf=d_rbf,
                                      d_state=SE3_param['l0_out_features'],
                                      d_hidden=d_hidden, p_drop=p_drop,
                                      symmetrize_repeats=symmetrize_repeats, repeat_length=repeat_length,
                                      symmsub_k=symmsub_k, sym_method=sym_method, main_block=main_block)

        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, d_rbf=d_rbf,
                               d_state=SE3_param['l0_out_features'],
                               SE3_param=SE3_param,
                               p_drop=p_drop,
                               nextra_l0=nextra_l0,
                               nextra_l1=nextra_l1)

    def forward(
        self, t_emb, msa, pair, xyz, state, idx, bond_feats, same_chain, is_motif, dist_matrix,
        sm_mask, is_protein, is_atomize_protein,
        symmids=None, symmsub=None, symmRs=None, symmmeta=None,
        use_checkpoint=False, top_k=128, atom_frames=None, extra_l0=None, extra_l1=None, use_atom_frames=False,
        crop=-1
    ):
        cas = xyz[:, :, 1].contiguous()  # 变成在内存连续的张量
        rbf_feat = util_module.rbf(torch.cdist(cas, cas)) + \
                   self.pos(idx, bond_feats, dist_matrix, sm_mask, is_protein, is_atomize_protein, same_chain)
        if use_checkpoint:
            msa = checkpoint.checkpoint(util_module.create_custom_forward(self.msa2msa), msa, pair, rbf_feat, state, use_reentrant=True)
            pair = checkpoint.checkpoint(util_module.create_custom_forward(self.msa2pair), msa, pair, use_reentrant=True)
            pair = checkpoint.checkpoint(util_module.create_custom_forward(self.pair2pair), pair, rbf_feat, state, crop, use_reentrant=True)

            xyz, state, alpha = checkpoint.checkpoint(util_module.create_custom_forward(self.str2str, top_k=top_k),
                                                      t_emb, msa.float(), pair.float(), xyz.detach().float(),
                                                      state.float(), idx, sm_mask, bond_feats, dist_matrix, atom_frames,
                                                      is_atomize_protein, extra_l0, extra_l1, use_atom_frames, use_reentrant=True)
            
        else:
            msa = self.msa2msa(msa, pair, rbf_feat, state)
            pair = self.msa2pair(msa, pair)
            pair = self.pair2pair(pair, rbf_feat, state, crop)
            
            xyz, state, alpha = self.str2str(
                t_emb, msa.float(), pair.float(), xyz.detach().float(), state.float(),
                idx, sm_mask, bond_feats, dist_matrix, atom_frames, is_atomize_protein, extra_l0, extra_l1, use_atom_frames,
                top_k=top_k, use_checkpoint=use_checkpoint
            )

        # update contacting subunits
        # symmetrize pair features
        # if symmsub is not None and symmsub.shape[0]>1:
        #     xyz, pair, symmsub = update_symm_subs(xyz, pair, symmids, symmsub, symmRs, symmmeta, self.fit, self.tscale)

        return msa, pair, xyz, state, alpha, symmsub


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4, n_finetune_block=0,
         d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32, d_time_emb=128, nextra_l0=0,
         d_mol_props=0,
         n_head_msa=8, n_head_pair=4,
         SE3_param={}, SE3_ref_param={}, p_drop=0.15,
         atom_type_index=None, aamask=None, 
         ljlk_parameters=None, lj_correction_parameters=None,
         cb_len=None, cb_ang=None, cb_tor=None,
         num_bonds=None, lj_lin=0.6, use_same_chain=False, enable_same_chain=False,
         use_chiral_l1=False, use_lj_l1=False, refiner_topk=64,
         symmetrize_repeats=None,
         repeat_length=None,
         symmsub_k=None,
         sym_method=None,
         main_block=None,
         fit=False,
         tscale=1.0
    ):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block
        self.n_finetune_block = n_finetune_block

        self.atom_type_index = atom_type_index
        self.aamask = aamask
        self.ljlk_parameters = ljlk_parameters 
        self.lj_correction_parameters = lj_correction_parameters
        self.num_bonds = num_bonds
        self.lj_lin = lj_lin
        self.cb_len = cb_len
        self.cb_ang = cb_ang
        self.cb_tor = cb_tor
        self.use_chiral_l1 = use_chiral_l1
        self.use_lj_l1 = use_lj_l1
        self.enable_same_chain = enable_same_chain
        self.fit = fit
        self.tscale = tscale
        self.d_time_emb = d_time_emb
        self.d_mol_props = d_mol_props

        self.alpha_embedder = nn.Linear(30, SE3_param['l0_out_features'])  # 扭转角编码
        if d_mol_props > 0:
            self.mol_prop_embedder = nn.Sequential(
                nn.Linear(d_mol_props, d_time_emb),
                ShiftedSoftplus(),
                nn.Linear(d_time_emb, d_time_emb),
            )
        else:
            self.mol_prop_embedder = None

        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        nextra_l0=nextra_l0,
                                                        p_drop=p_drop,
                                                        use_global_attn=True,
                                                        SE3_param=SE3_param,
                                                        nextra_l1=3 if self.use_chiral_l1 else 0,
                                                        use_same_chain=use_same_chain,
                                                        enable_same_chain=enable_same_chain,
                                                        symmetrize_repeats=symmetrize_repeats,
                                                        repeat_length=repeat_length,
                                                        symmsub_k=symmsub_k,
                                                        sym_method=sym_method,
                                                        main_block=main_block,
                                                        fit=fit,
                                                        tscale=tscale)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       nextra_l0=nextra_l0,
                                                       p_drop=p_drop,
                                                       use_global_attn=False,
                                                       SE3_param=SE3_param,
                                                       nextra_l1=3 if self.use_chiral_l1 else 0,
                                                       use_same_chain=use_same_chain,
                                                       enable_same_chain=enable_same_chain,
                                                       symmetrize_repeats=symmetrize_repeats,
                                                       repeat_length=repeat_length,
                                                       symmsub_k=symmsub_k,
                                                       sym_method=sym_method,
                                                       main_block=main_block,
                                                       fit=fit,
                                                       tscale=tscale)
                                                       for i in range(n_main_block)])

        # Final SE(3) refinement
        if n_ref_block > 0:
            n_extra_l1 = 0
            if self.use_chiral_l1:
                n_extra_l1 += 3
            if self.use_lj_l1:
                nextra_l0 += 2 * chemical.NTOTALDOFS
                n_extra_l1 += 3
            self.str_refiner = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                       d_state=SE3_param['l0_out_features'],
                                       SE3_param=SE3_ref_param,
                                       p_drop=p_drop,
                                       nextra_l0=nextra_l0,
                                       nextra_l1=n_extra_l1,
                                       )

        # To get all-atom coordinates
        # self.xyzconverter = util_module.XYZConverter()

        self.refiner_topk = refiner_topk

        self.reset_parameter()

    def reset_parameter(self):
        self.alpha_embedder = util_module.init_lecun_normal(self.alpha_embedder)
        nn.init.zeros_(self.alpha_embedder.bias)
        if self.mol_prop_embedder is not None:
            self.mol_prop_embedder[0] = util_module.init_lecun_normal(self.mol_prop_embedder[0])
            nn.init.zeros_(self.mol_prop_embedder[0].bias)
            nn.init.zeros_(self.mol_prop_embedder[2].weight)
            nn.init.zeros_(self.mol_prop_embedder[2].bias)

    def forward(
        self, t, msa, msa_full, pair, xyz, alpha, state, idx,
        symmids, symmsub, symmRs, symmmeta, 
        bond_feats, dist_matrix, same_chain, chirals, is_motif, sm_mask, is_protein, is_atomize_protein,
        atom_frames=None, use_checkpoint=False, use_atom_frames=False,
        p2p_crop=-1, topk_crop=0, mol_props=None
    ):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)

        if self.enable_same_chain == False:
            same_chain = None

        B, _, L = msa.shape[:3]
        # if symmsub is not None:
        #     Lasu = L//symmsub.shape[0]
        # else:
        #     Lasu = L
        xyz_s = list()
        alpha_s = list()

        """与RFDiffusion不同，本项目利用l0特征（该特征对旋转不变）传入时间步嵌入和分子性质嵌入"""
        t_emb = get_time_embedding(t, self.d_time_emb)[:, None, :].repeat(1, L, 1)
        if self.mol_prop_embedder is not None:
            if mol_props is None:
                mol_props = torch.zeros((B, self.d_mol_props), device=t_emb.device, dtype=t_emb.dtype)
            mol_props = torch.nan_to_num(mol_props.to(device=t_emb.device, dtype=t_emb.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            prop_emb = self.mol_prop_embedder(mol_props.float()).to(t_emb.dtype)
            # 分子性质只作为配体生成条件注入；蛋白部分保持纯时间步嵌入。
            ligand_prop_mask = sm_mask & ~is_atomize_protein
            t_emb = t_emb + prop_emb[:, None, :] * ligand_prop_mask[..., None].to(t_emb.dtype)
        extra_l0 = None

        extra_l1 = None

        alpha = alpha.view(B, L, -1)
        alpha = alpha * (~torch.where(is_atomize_protein, True, sm_mask)[..., None]).float()  # 配体部分（包括原子化残基）的扭转角置0
        alpha_emb = self.alpha_embedder(alpha)  # (B, L, d_state)
        state = state + alpha_emb  # 注入插值后的扭转角特征

        for i_m in range(self.n_extra_block):
            """配体生成任务并没有一个明确的手性信息，因此use_chiral_l1默认为False"""
            if self.use_chiral_l1:
                dchiraldxyz, = calc_chiral_grads(xyz.detach(),chirals)
                extra_l1 = dchiraldxyz[0].detach()
                
            msa_full, pair, xyz, state, alpha, symmsub = self.extra_block[i_m](t_emb, msa_full, pair,
                                                               xyz, state, idx, bond_feats, same_chain,
                                                               is_motif, dist_matrix, sm_mask, is_protein, is_atomize_protein,
                                                               use_checkpoint=use_checkpoint,
                                                               top_k=topk_crop, atom_frames=atom_frames,
                                                               extra_l0=extra_l0, extra_l1=extra_l1,
                                                               use_atom_frames=use_atom_frames, crop=p2p_crop)
            xyz_s.append(xyz)
            alpha_s.append(alpha)

        for i_m in range(self.n_main_block):
            extra_l1 = None
            """配体生成任务并没有一个明确的手性信息"""
            if self.use_chiral_l1:
                dchiraldxyz, = calc_chiral_grads(xyz.detach(),chirals)
                extra_l1 = dchiraldxyz[0].detach()
            msa, pair, xyz, state, alpha, symmsub = self.main_block[i_m](t_emb, msa, pair,
                                                         xyz, state, idx, bond_feats, same_chain,
                                                         is_motif, dist_matrix, sm_mask, is_protein, is_atomize_protein,
                                                         use_checkpoint=use_checkpoint,
                                                         top_k=topk_crop, atom_frames=atom_frames,
                                                         extra_l0=extra_l0, extra_l1=extra_l1,
                                                         use_atom_frames=use_atom_frames, crop=p2p_crop)
            xyz_s.append(xyz)
            alpha_s.append(alpha)

        # _, xyzallatom = self.xyzconverter.compute_all_atom(seq, xyz, alpha[:, :, :, :2])  # think about detach here...

        """重复调用时不要启用checkpoint，否则ddp会无法接收梯度"""
        for i_m in range(self.n_ref_block):
            extra_l0 = []
            extra_l1 = []

            """由于直到生成的最后一步前，配体的拓扑结构是未确定的，因此强烈建议不启用lj_l1"""
            # if self.use_lj_l1:
            #     dljdxyz, dljdalpha = calc_lj_grads(seq, xyz.detach(), input_alpha.detach(), self.xyzconverter.compute_all_atom,
            #          bond_feats, dist_matrix, self.aamask, self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
            #          lj_lin=self.lj_lin)
            #     extra_l0.append(dljdalpha.reshape(1,-1,2 * chemical.NTOTALDOFS).detach())
            #     extra_l1.append(dljdxyz[0].detach())


            if self.use_chiral_l1:
                dchiraldxyz, = calc_chiral_grads(xyz.detach(),chirals)
                #extra_l1 = torch.cat((dljdxyz[0].detach(), dchiraldxyz[0].detach()), dim=1)
                extra_l1.append(dchiraldxyz[0].detach())
            if len(extra_l0) > 0:
                extra_l0 = torch.concat(extra_l0, dim=-1)
            else:
                extra_l0 = None
            if len(extra_l1) > 0:
                extra_l1 = torch.cat(extra_l1, dim=1)
            else:
                extra_l1 = None

            xyz, state, alpha = self.str_refiner(t_emb, msa.float(), pair.float(), xyz.detach().float(), state.float(),
                                                 idx, sm_mask, bond_feats, dist_matrix, atom_frames, is_atomize_protein,
                                                 extra_l0, extra_l1, top_k=self.refiner_topk,
                                                 use_atom_frames=use_atom_frames)

            # if symmsub is not None and symmsub.shape[0]>1:
            #     xyz = update_symm_Rs(xyz, Lasu, symmsub, symmRs, self.fit, self.tscale)

            xyz_s.append(xyz)
            alpha_s.append(alpha)
        
        # _, xyzallatom = self.xyzconverter.compute_all_atom(seq, xyz, alpha[:, :, :, :2])  # think about detach here...
        # xyzallatom_s = list()
        # xyzallatom_s.append(xyzallatom.clone())

        xyz = torch.stack(xyz_s, dim=0)
        alpha_s = torch.stack(alpha_s, dim=0)
        # xyzallatom_s = torch.cat(xyzallatom_s, dim=0)

        return msa, pair, xyz, alpha_s, state
