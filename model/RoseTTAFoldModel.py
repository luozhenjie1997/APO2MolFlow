import torch
import torch.nn as nn
from icecream import ic
from .layers.Embeddings import MSA_emb, Extra_emb, Bond_emb, Templ_emb, recycling_factory
from .Track_module import IterativeSimulator
from .layers.AuxiliaryPredictor import (
    DistanceNetwork,
    MaskedTokenNetwork,
    BondTypeNetwork,
    LDDTNetwork,
    PAENetwork,
    BinderNetwork,
)
from .utils import chemical
from .utils.utils import get_time_embedding

def get_shape(t):
    if hasattr(t, "shape"):
        return t.shape
    if type(t) is tuple:
        return [get_shape(e) for e in t]
    else:
        return type(t)


class RoseTTAFoldModule(nn.Module):
    def __init__(
        self, 
        symmetrize_repeats=None,       # whether to symmetrize repeats in the pair track 
        repeat_length=None,            # if symmetrizing repeats, what length are they? 
        symmsub_k=None,                # if symmetrizing repeats, which diagonals?
        sym_method=None,               # if symmetrizing repeats, which block symmetrization method? 
        main_block=None,               # if copying template blocks along main diag, which block is main block? (the one w/ motif)
        copy_main_block_template=None, # whether or not to copy main block template along main diag
        n_extra_block=4, 
        n_main_block=8, 
        n_ref_block=4, 
        n_finetune_block=0,
        d_msa=256, 
        d_msa_full=64, 
        d_pair=128, 
        d_templ=64,
        n_head_msa=8, 
        n_head_pair=4, 
        n_head_templ=4,
        d_hidden=32, 
        d_hidden_templ=64,
        d_t1d=0,
        d_t2d=0,
        d_time_emb=128,
        d_mol_props=0,
        nextra_l0=0,
        p_drop=0.15,
        additional_dt1d=0,
        recycling_type="msa_pair",
        SE3_PARAM={}, SE3_REF_PARAM={},
        atom_type_index=None, 
        aamask=None, 
        ljlk_parameters=None, 
        lj_correction_parameters=None, 
        cb_len=None, 
        cb_ang=None, 
        cb_tor=None,
        num_bonds=None, 
        lj_lin=0.6, 
        use_chiral_l1=False,
        use_lj_l1=False,
        use_atom_frames=False,
        use_same_chain=True,
        enable_same_chain=True,
        refiner_topk=64,
        # New for diffusion
        freeze_track_motif=False,
        assert_single_sequence_input=False,
        fit=False,
        tscale=1.0
    ):
        super(RoseTTAFoldModule, self).__init__()
        self.freeze_track_motif = freeze_track_motif
        self.assert_single_sequence_input = assert_single_sequence_input
        self.recycling_type = recycling_type
        self.d_time_emb = d_time_emb
        #
        # Input Embeddings
        d_state = SE3_PARAM["l0_out_features"]
        self.latent_emb = MSA_emb(
            d_msa=d_msa, d_pair=d_pair, d_state=d_state, p_drop=p_drop, use_same_chain=use_same_chain,
            enable_same_chain=enable_same_chain
        )
        self.full_emb = Extra_emb(
            d_msa=d_msa_full, d_init=chemical.NAATOKENS + 2, p_drop=p_drop
        )
        self.bond_emb = Bond_emb(d_pair=d_pair, d_init=chemical.NBTYPES)

        self.templ_emb = Templ_emb(d_t1d=d_t1d,
                                   d_t2d=d_t2d,
                                   d_pair=d_pair,
                                   d_templ=d_templ, 
                                   d_state=d_state, 
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ,
                                   d_time_emb=None,
                                   p_drop=0.25,
                                   symmetrize_repeats=symmetrize_repeats, # repeat protein stuff 
                                   repeat_length=repeat_length, 
                                   symmsub_k=symmsub_k,
                                   sym_method=sym_method, 
                                   main_block=main_block, 
                                   copy_main_block=copy_main_block_template,
                                   additional_dt1d=additional_dt1d)

        # Update inputs with outputs from previous round

        self.recycle = recycling_factory[recycling_type](d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block,
            n_main_block=n_main_block,
            n_ref_block=n_ref_block,
            n_finetune_block=n_finetune_block,
            d_msa=d_msa,
            d_msa_full=d_msa_full,
            d_pair=d_pair,
            d_hidden=d_hidden,
            d_time_emb=d_time_emb,
            d_mol_props=d_mol_props,
            nextra_l0=nextra_l0,
            n_head_msa=n_head_msa,
            n_head_pair=n_head_pair,
            SE3_param=SE3_PARAM,
            SE3_ref_param=SE3_REF_PARAM,
            p_drop=p_drop,
            atom_type_index=atom_type_index,  # change if encoding elements instead of atomtype
            aamask=aamask,
            ljlk_parameters=ljlk_parameters,
            lj_correction_parameters=lj_correction_parameters,
            num_bonds=num_bonds,
            cb_len=cb_len,
            cb_ang=cb_ang,
            cb_tor=cb_tor,
            lj_lin=lj_lin,
            use_lj_l1=use_lj_l1,
            use_chiral_l1=use_chiral_l1,
            symmetrize_repeats=symmetrize_repeats,
            repeat_length=repeat_length,
            symmsub_k=symmsub_k,
            sym_method=sym_method,
            main_block=main_block,
            use_same_chain=use_same_chain,
            enable_same_chain=enable_same_chain,
            refiner_topk=refiner_topk
        )

        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.bond_pred = BondTypeNetwork(d_pair, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
        # self.pae_pred = PAENetwork(d_pair)
        # self.pde_pred = PAENetwork(d_pair)  # 距离误差(PDE)，但采用与对齐误差相同的架构。
        # binder predictions are made on top of the pair features, just like
        # PAE predictions are. It's not clear if this is the best place to insert
        # this prediction head.
        # self.binder_network = BinderNetwork(d_pair, d_state)

        # self.bind_pred = BinderNetwork() #fd - expose n_hidden as variable?

        self.use_atom_frames = use_atom_frames
        self.enable_same_chain = enable_same_chain

    def forward(self, t, msa_latent, msa_full, seq, xyz, alpha, idx, bond_feats, dist_matrix, sm_mask, is_atomize_protein,
                chirals=None, sctors=None, seq1hot=None, bond_noisy=None, atom_frames=None, t1d=None, t2d=None, xyz_t=None, alpha_t=None, mask_t=None, same_chain=None,
                mol_props=None,
                msa_prev=None, pair_prev=None, state_prev=None, mask_recycle=None, is_motif=None,
                return_raw=False, is_protein=None,
                use_checkpoint=False,
                p2p_crop=-1, topk_crop=-1,   # striping
                symmids=None, symmsub=None, symmRs=None, symmmeta=None,  # symmetry
    ):
        B, N, L = msa_latent.shape[:3]
        dtype = msa_latent.dtype

        msa_latent, pair, state = self.latent_emb(
            msa_latent, seq, idx, bond_feats, dist_matrix, sm_mask, is_protein, is_atomize_protein, same_chain=same_chain,
            seq1hot=seq1hot, bond_noisy=bond_noisy
        )
        msa_full = self.full_emb(msa_full, seq, seq1hot=seq1hot)
        pair = pair + self.bond_emb(bond_feats)

        msa_latent, pair, state = msa_latent.to(dtype), pair.to(dtype), state.to(dtype)  # 确保数据类型一致
        msa_full = msa_full.to(dtype)  # 确保数据类型一致

        #
        # Do recycling
        if msa_prev is None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
        if pair_prev is None:
            pair_prev = torch.zeros_like(pair)
        if state_prev is None or self.recycling_type == "msa_pair":  # explicitly remove state features if only recycling msa and pair
            state_prev = torch.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(msa_prev, pair_prev, xyz, state_prev, sctors, mask_recycle)
        msa_recycle, pair_recycle = msa_recycle.to(dtype), pair_recycle.to(dtype)  # 确保数据类型一致

        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle  # 如果状态未被回收，这些将为零

        # 添加模板特征
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, mask_t, pair, state, use_checkpoint=use_checkpoint, p2p_crop=p2p_crop)

        # 根据给定输入预测坐标更新量
        is_motif = is_motif if self.freeze_track_motif else torch.zeros_like(seq).bool()[0]
        msa, pair, xyz, alpha_s, state = self.simulator(
            t, msa_latent, msa_full, pair, xyz[:,:,:3], alpha, state, idx,
            symmids, symmsub, symmRs, symmmeta,
            bond_feats, dist_matrix, same_chain, chirals, is_motif, sm_mask, is_protein, is_atomize_protein, atom_frames,
            use_checkpoint=use_checkpoint, use_atom_frames=self.use_atom_frames, 
            p2p_crop=p2p_crop, topk_crop=topk_crop, mol_props=mol_props
        )

        if return_raw:
            # get last structure
            xyz_last = xyz[-1].unsqueeze(0)
            return msa[:,0], pair, state, xyz_last, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)

        # predict distogram & orientograms
        logits = self.c6d_pred(pair)

        # 化学键预测
        bond_logits_symm = self.bond_pred(pair)

        # Predict LDDT
        lddt = self.lddt_pred(state)

        # logits_pae = logits_pde = p_bind = None
        # predict aligned error and distance error
        # logits_pae = self.pae_pred(pair)
        # logits_pde = self.pde_pred(pair + pair.permute(0, 2, 1, 3))  # symmetrize pair features

        #fd  predict bind/no-bind
        # p_bind = self.bind_pred(logits_pae, same_chain)

        return (
            logits, logits_aa, bond_logits_symm, xyz, alpha_s, lddt
        )
