import os
import torch
import random
import torch.nn.functional as F
import time
import torch.distributed as dist
import model.utils.util_module as util_module
import model.utils.utils as utils
import model.loss.loss as loss
import apex
from model.data.data_loader import Apo2HoloDataset
from model.RoseTTAFoldModel import RoseTTAFoldModule
from model.utils.interpolant import Interpolant
from openfold.all_atom import to_atom37
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from contextlib import nullcontext
from model.utils.scheduler import get_stepwise_decay_schedule_with_warmup
from model.utils.chemical import allatom_mask, atom_type_index, ljlk_parameters, lj_correction_parameters, num_bonds, \
    cb_length_t, cb_angle_t, cb_torsion_t, NBTYPES, NTOTAL

"""
训练代码均以mini_batch=1的情况进行编写
"""

USE_GEOMETRY = False  # 是否启用几何损失

# torch.autograd.set_detect_anomaly(True)
# 解析命令行参数（通过torchrun启动时会自动设置环境变量）
LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # 当前GPU的本地rank
WORLD_SIZE = int(os.environ['WORLD_SIZE'])  # 总GPU数


# 初始化分布式训练环境
def setup_distributed():
    dist.init_process_group(
        backend="nccl",  # NVIDIA GPU推荐使用NCCL后端
        init_method="env://",
        rank=LOCAL_RANK,
        world_size=WORLD_SIZE
    )
    torch.cuda.set_device(LOCAL_RANK)  # 绑定当前进程到对应GPU


def train(local_rank):
    config, config_name = utils.load_config('./config/base_config.yaml')
    loss_weight = config.train.loss_weight

    utils.set_seed(config.train.seed + local_rank)
    accumulation_steps = config.train.pseudo_batch_size // config.train.mini_batch_size
    # loss_weight = config.train.loss_weight

    setup_distributed()  # 初始化分布式环境

    # 初始化模型
    model = RoseTTAFoldModule(**config.model.params, aamask=allatom_mask, atom_type_index=atom_type_index,
                              ljlk_parameters=ljlk_parameters, lj_correction_parameters=lj_correction_parameters,
                              num_bonds=num_bonds, cb_len=cb_length_t, cb_ang=cb_angle_t, cb_tor=cb_torsion_t).to(local_rank)
    # 为不同的网络层设置相应的权重衰减
    param_group = util_module.add_weight_decay(model, l2_coeff=config.train.l2_coeff)
    optimizer = apex.optimizers.FusedAdam(param_group, lr=config.train.lr, set_grad_none=True)
    scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 2000, 0.95)

    interpolant = Interpolant(device=local_rank).to(local_rank)  # 用于对数据进行插值
    # 计算所有原子坐标的工具
    atom_coords_converter = util_module.XYZConverter().to(local_rank)

    if os.path.exists('./save_model/checkpoint.ckpt'):
        checkpoint = torch.load('./save_model/checkpoint.ckpt', map_location='cuda:%s' % local_rank)
        load_info = model.load_state_dict(checkpoint['last_state_dict'], strict=False)
        if local_rank == 0 and (len(load_info.missing_keys) > 0 or len(load_info.unexpected_keys) > 0):
            print("Model checkpoint loaded with missing keys: %s, unexpected keys: %s" %
                  (load_info.missing_keys, load_info.unexpected_keys))
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as exc:
            if local_rank == 0:
                print("Skip optimizer state loading because model parameters changed: %s" % exc)
        epoch = checkpoint['epoch']
        log_name = checkpoint['log_name']
        log_steps = checkpoint['log_steps']
        # random.setstate(checkpoint['random_state'])
        # torch.set_rng_state(checkpoint['rng_state'])  # 恢复CPU随机状态
        # torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])  # 恢复所有GPU的随机状态
        del checkpoint
    else:
        model = utils.load_rfaa_weights_without_nucleic_acids(model, config['train']['rfaa_weight_pth'])  # 继承RFAA的权重
        epoch = 0
        log_steps = 1
        log_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, gradient_as_bucket_view=True)  # DDP包装
    dist.barrier()

    dataset = Apo2HoloDataset(base_path=config['dataset']['base_path'], apo_path=config['dataset']['apo_path'],
                              holo_path=config['dataset']['holo_path'], holo_list_path=config['dataset']['base_path'] + '/use_holo_ids_filter_mw.pkl',
                              n_crop=config['train']['n_crop'], atomize_protein=config['model']['atomize_protein'],
                              crop_strategy=config['train']['crop_strategy'])
    sampler = DistributedSampler(dataset, shuffle=True, num_replicas=WORLD_SIZE, rank=local_rank)  # 分布式采样器

    batch_size = config.train.batch_size
    data_total = len(dataset)

    # num_workers = 4
    num_workers = config.train.num_workers
    dataloader = DataLoader(dataset, batch_size=config.train.mini_batch_size, pin_memory=True, persistent_workers=True,
                            num_workers=num_workers, prefetch_factor=config.train.pseudo_batch_size, sampler=sampler)

    if local_rank == 0:
        writer = SummaryWriter(log_dir='./logs/%s' % log_name)
        print("Start time:%s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    else:
        writer = None

    n_cycle = config.model.n_cycle

    for e in range(epoch, config.train.epochs):
        sampler.set_epoch(e)  # 每个epoch重置采样器
        model.train()

        used_count = 0
        done_flag = torch.tensor(0, device=local_rank)  # 用于控制所有进程是否直接进入下一个epoch

        batch_total_loss = torch.tensor(0., device=local_rank)
        batch_sm_token_loss = torch.tensor(0., device=local_rank)
        batch_bond_loss = torch.tensor(0., device=local_rank)
        batch_protein_sm_fape_loss = torch.tensor(0., device=local_rank)
        batch_sm_protein_fape_loss = torch.tensor(0., device=local_rank)
        batch_protein_trans_loss = torch.tensor(0., device=local_rank)
        batch_protein_rots_loss = torch.tensor(0., device=local_rank)
        batch_tors_loss = torch.tensor(0., device=local_rank)
        batch_protein_blen_loss = torch.tensor(0., device=local_rank)
        batch_protein_bang_loss = torch.tensor(0., device=local_rank)
        batch_ligand_coords_loss = torch.tensor(0., device=local_rank)
        batch_protein_ligand_contact_loss = torch.tensor(0., device=local_rank)
        batch_ligand_bond_loss = torch.tensor(0., device=local_rank)
        batch_ligand_rigid_loss = torch.tensor(0., device=local_rank)
        batch_ligand_chiral_loss = torch.tensor(0., device=local_rank)
        n_ligand_chiral = torch.tensor(0., device=local_rank)
        batch_c6d_loss = torch.tensor(0., device=local_rank)
        batch_lddt = torch.tensor(0., device=local_rank)
        batch_plddt = torch.tensor(0., device=local_rank)
        batch_plddt_loss = torch.tensor(0., device=local_rank)

        # 仅主进程显示进度条
        if local_rank == 0:
            pbar = tqdm(range(len(dataloader)), dynamic_ncols=True, leave=False)
            pbar.set_description_str("epoch:%s" % (e + 1))
        else:
            pbar = None

        for i, pdb_data in enumerate(dataloader):
            holo_pdb_id, apo_pdb_id = pdb_data['holo_pdb_id'], pdb_data['apo_pdb_id']
            # TODO: 以下逻辑只限定在批次为1的情况，后续需要修正
            is_predict = pdb_data['is_predict'][0]
            pdb_data = utils.recursive_to(pdb_data, local_rank)

            # if i == 466 and local_rank == 1:
            #     print(holo_pdb_id, apo_pdb_id)
            #     exit(-1)
            # else:
            #     continue

            B, L = pdb_data['idx'].shape[:2]
            mol_props = pdb_data.get('condition', {}).get('mol_props', None)

            if local_rank == 0:
                pbar.set_description_str("epoch:%s, L=%s" % (e + 1,L))

            mask_protein_BB = pdb_data['mask_protein_BB']

            xt = interpolant(pdb_data)  # 对需要的数据进行插值
            xyzs_prev = to_atom37(xt['trans_t'], xt['rots_t'])[:, :, :NTOTAL]  # 还原t时刻的主链坐标
            sm_xyzs_t = torch.concat([
                torch.zeros((B, L, 1, 3), device=xyzs_prev.device),
                torch.concat([torch.zeros((B, L - xt['ligand_xyzs_t'].shape[1], 1, 3), device=xyzs_prev.device),
                              xt['ligand_xyzs_t'].view(-1).reshape(B, xt['ligand_xyzs_t'].shape[1], 1, 3)], dim=1),
                torch.zeros((B, L, pdb_data['xyzs']['xyzs_mask'].shape[-1] - 2, 3), device=xyzs_prev.device)], dim=2)  # 将CA原子位置设置为配体坐标
            xyzs_prev = torch.where(pdb_data['is_protein'][:, :, None, None], xyzs_prev, sm_xyzs_t)  # 替换t时刻的配体坐标
            xyzs_prev = torch.where(pdb_data['is_atomize_protein'][:, :, None, None], sm_xyzs_t, xyzs_prev)  # 替换t时刻的原子化后的残基坐标
            # 将msa的配体部分替换为加噪后的数据
            pdb_data['seq']['msa_latent'][:, 0, :, :70] = xt['seq_xt']
            pdb_data['seq']['msa_latent'][:, 0, :, 70:140] = xt['seq_xt']
            pdb_data['seq']['msa_full'][:, 0, :, :70] = xt['seq_xt']
            # 将以一维模板特征的配体部分替换为加噪后的数据
            pdb_data['template']['t1d'][:, 0, :, :70] = xt['seq_xt']

            use_checkpoint = True
            # if L > 190:
            #     use_checkpoint = True
            # else:
            #     use_checkpoint = False

            sm_mask = pdb_data['sm_mask']
            sm_pair_mask = sm_mask[:, None, :] * sm_mask[:, :, None]

            # 计算循环使用的特征
            msa_prev = None
            pair_prev = None
            state_prev = None

            n = torch.randint(0, n_cycle - 1, (1,), device=local_rank)  # 若n=1，模型运行1次，实际上不会回收特征
            dist.broadcast(n, src=0)  # 同步特征回收次数
            # xyzs_prev = xyzs_prev * config['model']['position_scale']  # 坐标缩放
            with torch.no_grad():
                with model.no_sync():
                    for n in range(n):
                        msa_prev, pair_prev, state_prev, _, _ = model(
                            t=xt['t'], msa_latent=pdb_data['seq']['msa_latent'], msa_full=pdb_data['seq']['msa_full'], seq=xt['seq_token_xt'],
                            seq1hot=xt['seq_xt'], bond_noisy=xt['bond_xt_logtis'], xyz=xyzs_prev, alpha=xt['chi_t'], idx=pdb_data['idx'], bond_feats=xt['bond_xt'],
                            dist_matrix=pdb_data['dist_matrix'], t1d=pdb_data['template']['t1d'], t2d=pdb_data['template']['t2d'],
                            alpha_t=pdb_data['template']['alpha_t'], xyz_t=pdb_data['template']['xyz_t'][:, :, :, 1],
                            mask_t=pdb_data['template']['mask_t_2d'], sm_mask=sm_mask, is_protein=pdb_data['is_protein'],
                            is_atomize_protein=pdb_data['is_atomize_protein'], atom_frames=pdb_data['atom_frames'],
                            same_chain=pdb_data['same_chain'], msa_prev=msa_prev, pair_prev=pair_prev, state_prev=state_prev,
                            mol_props=mol_props,
                            return_raw=True, topk_crop=config['model']['topk_crop'], p2p_crop=config['train']['p2p_crop'])

            # 在单机多卡环境中，只在每个进程的最后一个mini-batch中进行梯度同步
            my_context = model.no_sync if local_rank != -1 and used_count < (config.train.pseudo_batch_size - 1) else nullcontext

            with autocast(enabled=True, dtype=torch.bfloat16):
                with my_context():
                    logits_c6d, logits_aa, bond_logits_symm, xyz, alpha_s, logits_plddt = model(
                        t=xt['t'], msa_latent=pdb_data['seq']['msa_latent'], msa_full=pdb_data['seq']['msa_full'],
                        seq=xt['seq_token_xt'], seq1hot=xt['seq_xt'], bond_noisy=xt['bond_xt_logtis'], xyz=xyzs_prev, alpha=xt['chi_t'], idx=pdb_data['idx'],
                        bond_feats=xt['bond_xt'], dist_matrix=pdb_data['dist_matrix'], t1d=pdb_data['template']['t1d'],
                        t2d=pdb_data['template']['t2d'], alpha_t=pdb_data['template']['alpha_t'], xyz_t=pdb_data['template']['xyz_t'][:, :, :, 1],
                        mask_t=pdb_data['template']['mask_t_2d'], sm_mask=sm_mask, is_protein=pdb_data['is_protein'],
                        is_atomize_protein=pdb_data['is_atomize_protein'], atom_frames=pdb_data['atom_frames'],
                        same_chain=pdb_data['same_chain'], use_checkpoint=use_checkpoint, msa_prev=msa_prev, pair_prev=pair_prev,
                        state_prev=state_prev, mol_props=mol_props, topk_crop=config['model']['topk_crop'], p2p_crop=config['train']['p2p_crop'])
                # xyz = xyz / config['model']['position_scale']  # 解除坐标缩放
                I, B, L, _, _ = xyz.shape

                # TODO: 以下损失缩放逻辑只限定在批次为1的情况，后续需要修正
                # 根据时间步信息创建的损失缩放，时间越早（小），损失越高
                norm_scale = 1 - torch.min(xt['t'][..., None], torch.tensor(config['model']['interpolant']['t_normalization_clip']))[0, 0]  # (B, 1, 1)

                """配体原子类型预测损失"""
                logits_aa = logits_aa.permute(0, 2, 1)
                sm_token_loss = F.cross_entropy(logits_aa.view(-1, logits_aa.shape[-1]), pdb_data['seq']['seq'].view(-1), reduction='none')
                sm_token_loss = ((sm_token_loss * sm_mask.float().reshape(-1)).reshape(logits_aa.shape[0], logits_aa.shape[1]).sum(1) /
                                 (sm_mask.sum(1) + 1e-8)).mean()
                """化学键类型损失，只计算上（下）半矩阵即可"""
                # 只选取配体部分
                ligand_len = sm_mask.sum().item()
                bond_logits_symm = bond_logits_symm[sm_pair_mask].reshape(B, ligand_len, ligand_len, NBTYPES)  # 只计算配体部分的化学键
                bond_true = pdb_data['bond_feats'][sm_pair_mask].reshape(B, ligand_len, ligand_len)
                tri_mask = torch.triu(torch.ones(ligand_len, ligand_len, device=local_rank), diagonal=1).unsqueeze(0)  # 构造上三角矩阵掩码。diagonal=1会将对角线设置为0
                # 只选取需要计算部分
                bond_logits_symm = bond_logits_symm[tri_mask.bool()]
                bond_true = bond_true[tri_mask.bool()]
                bond_loss = loss.focal_loss_multiclass(bond_logits_symm, bond_true, gamma=3.)

                predRs_all, pred_allatom = atom_coords_converter.compute_all_atom(pdb_data['seq']['seq'], xyz[-1], alpha_s[-1])
                natRs_all, _n0 = atom_coords_converter.compute_all_atom(pdb_data['seq']['seq'], pdb_data['xyzs']['xyzs_true'][..., :3],
                                                                     pdb_data['xyzs']['torsion_angle_true'], non_ideal=True)
                natRs_all_alt, _n1 = atom_coords_converter.compute_all_atom(pdb_data['seq']['seq'], pdb_data['xyzs']['xyzs_true'][..., :3],
                                                                     pdb_data['xyzs']['torsion_angle_alt_true'], non_ideal=True)
                # 解决对称性问题
                natRs_all_symm, nat_symm = loss.resolve_symmetry(pred_allatom[-1], natRs_all[0], pdb_data['xyzs']['xyzs_true'][0],
                                                                 natRs_all_alt[0], pdb_data['xyzs']['xyzs_true_alt'][0], pdb_data['xyzs']['xyzs_mask'][0])
                # 配体部分不用进行对称性处理
                nat_symm[~mask_protein_BB[0], :, :3] = pdb_data['xyzs']['xyzs_true'][:, ~mask_protein_BB[0], :, :3]

                """蛋白质主链框架损失。使用RFDiffusion提出的框架损失"""
                # 获取预测坐标的框架
                N_pred, Ca_pred, C_pred = xyz[:, :, :, 0], xyz[:, :, :, 1], xyz[:, :, :, 2]
                R_pred, T_pred = utils.rigid_from_3_points(N_pred.reshape(I * B, L, 3), Ca_pred.reshape(I * B, L, 3), C_pred.reshape(I * B, L, 3))
                R_pred = R_pred.reshape(I, B, L, 3, 3)
                T_pred = T_pred.reshape(I, B, L, 3)
                protein_frame_distance_loss, trans_err, rots_err = loss.frame_distance_loss(R_pred[:, :, pdb_data['is_protein'][0]],
                                                                                            T_pred[:, :, pdb_data['is_protein'][0]],
                                                                                            pdb_data['xyzs']['xyzs_true_rots'][:, pdb_data['is_protein'][0]],
                                                                                            pdb_data['xyzs']['xyzs_true_trans'][:, pdb_data['is_protein'][0]],
                                                                                            mask_protein_BB[:, pdb_data['is_protein'][0]],
                                                                                            gamma=config.train.loss_weight.gamma)

                """蛋白质扭转角损失"""
                tors_loss = loss.torsionAngleLoss(alpha_s, pdb_data['xyzs']['torsion_angle_true'], pdb_data['xyzs']['torsion_angle_alt_true'],
                                                  pdb_data['xyzs']['torsion_mask'], pdb_data['xyzs']['planar'])

                """配体坐标损失"""
                ligand_mask = torch.where(pdb_data['is_atomize_protein'], True, ~pdb_data['is_protein'])  # 需要包括原子化残基的配体掩码
                pred_ligand = xyz[:, ligand_mask, :, 1].reshape(I, B, -1, 3)  # (I, B, L, 3)
                true_ligand = pdb_data['xyzs']['xyzs_true'][ligand_mask, :, 1].reshape(B, -1, 3)
                ligand_coords_loss = loss.calc_ligand_coord_loss(pred_ligand, true_ligand, ligand_mask[ligand_mask].reshape(B, -1),
                                                                 gamma=config.train.loss_weight.gamma)

                """蛋白-配体接触距离损失"""
                protein_contact_mask = pdb_data['is_pocket_mask'] & pdb_data['is_protein'] & mask_protein_BB
                protein_ligand_contact_loss = loss.calc_protein_ligand_contact_loss(
                    pred_allatom, pdb_data['xyzs']['xyzs_true'], pdb_data['xyzs']['xyzs_mask'],
                    protein_contact_mask, ligand_mask
                )

                mask = (pdb_data['xyzs']['xyzs_mask'][:, :, 1] == 1.0)
                pair_mask = mask[:, None, :] * mask[..., None]

                """距离直方图损失"""
                c6d_loss = loss.calc_c6d_loss(logits_c6d, pdb_data['xyzs']['c6d'].long(), pair_mask)

                """lddt预测损失"""
                lddt, plddt, plddt_loss = loss.calc_allatom_lddt_loss(pred_allatom[:, :, :14].detach(), nat_symm[:, :14],
                                                                      logits_plddt, pdb_data['idx'], pdb_data['xyzs']['xyzs_mask'][:, :, :14],
                                                                      pair_mask, pdb_data['same_chain'], N_stripe=10)

                """pae预测损失"""
                # pae_loss = loss.calc_pae_loss(logits_pae, xyz[-1], pdb_data['xyzs']['xyzs_true'], mask, sm_mask, pdb_data['atom_frames'])

                if USE_GEOMETRY:
                    """蛋白质主链键长和键角损失"""
                    protein_blen_loss, protein_bang_loss = loss.calc_BB_bond_geom(pred_allatom, pdb_data['idx'], mask_protein_BB)

                    """配体化学键距离损失"""
                    atom_bond_loss, skip_bond_loss, rigid_loss = loss.calc_atom_bond_loss(xyz[-1], pdb_data['xyzs']['xyzs_true'][:, :, :3],
                                                                                          pdb_data['bond_feats'], pdb_data['seq']['seq'])

                    """配体手性损失"""
                    chiral_loss = loss.calc_chiral_loss(pred_ligand[-1], pdb_data['chirals'])
                    if pdb_data['chirals'].shape[1] != 0:  # 只有存在手性信息时才算做分母，避免在tensorboard中绘制的图像不稳定
                        n_ligand_chiral += 1
                        ligand_geometry_loss = atom_bond_loss + skip_bond_loss + rigid_loss + chiral_loss
                    else:
                        ligand_geometry_loss = atom_bond_loss + skip_bond_loss + rigid_loss
                    geometry_loss = (loss_weight.protein_geometry * (protein_blen_loss + protein_bang_loss) +
                                     loss_weight.ligand_geometry * ligand_geometry_loss)
                else:
                    geometry_loss = torch.tensor(0.0, device=xyz.device)

                # TODO: 以下辅助损失应用逻辑只限定在批次为1的情况，后续需要修正
                aux_loss = (loss_weight.protein_ligand_contact * protein_ligand_contact_loss + geometry_loss)
                if xt['t'].mean() < config['train']['use_aux_loss']:  # 时间步大于指定阈值时才启用辅助损失
                    aux_loss *= 0.

                total_loss = (loss_weight.sm_token * sm_token_loss + loss_weight.bond_token * bond_loss + loss_weight.tors * tors_loss / norm_scale +
                              loss_weight.frame_distance * protein_frame_distance_loss / norm_scale + loss_weight.ligand_coords * ligand_coords_loss / norm_scale +
                              loss_weight.c6d * c6d_loss + loss_weight.plddt * plddt_loss + aux_loss) / config.train.pseudo_batch_size
                if is_predict:
                    (total_loss * config['train']['predict_weight']).backward()
                else:
                    total_loss.backward()
            if torch.isnan(total_loss):
                print(holo_pdb_id, apo_pdb_id, 'loss is nan')
                exit(-1)

            batch_total_loss += total_loss.detach() / config.train.gpu_nums
            batch_sm_token_loss += sm_token_loss.detach()
            batch_bond_loss += bond_loss.detach()
            batch_protein_trans_loss += trans_err.detach()
            batch_protein_rots_loss += rots_err.detach()
            batch_tors_loss += tors_loss.detach()
            batch_ligand_coords_loss += ligand_coords_loss.detach()
            batch_protein_ligand_contact_loss += protein_ligand_contact_loss.detach()
            batch_c6d_loss += c6d_loss.detach()
            batch_lddt += lddt.detach()
            batch_plddt += plddt.detach()
            batch_plddt_loss += plddt_loss.detach()
            if USE_GEOMETRY:
                batch_protein_blen_loss += protein_blen_loss.detach()
                batch_protein_bang_loss += protein_bang_loss.detach()
                batch_ligand_bond_loss += (atom_bond_loss + skip_bond_loss).detach()
                batch_ligand_rigid_loss += rigid_loss.detach()
                batch_ligand_chiral_loss += chiral_loss.detach()
            used_count += 1

            if used_count % accumulation_steps == 0:
                # 梯度裁剪，并记录梯度的范数情况。max_norm=float('inf') 表示不进行裁剪，只计算范数。
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                if torch.isnan(grad_norm):
                    print(holo_pdb_id, apo_pdb_id, "grad is nan")
                    print([
                        batch_total_loss, batch_sm_token_loss, batch_bond_loss, batch_protein_sm_fape_loss,
                        batch_sm_protein_fape_loss, batch_protein_trans_loss, batch_protein_rots_loss,
                        batch_tors_loss, batch_protein_blen_loss, batch_protein_bang_loss, batch_ligand_coords_loss,
                        batch_protein_ligand_contact_loss, batch_ligand_bond_loss, batch_ligand_rigid_loss,
                        batch_ligand_chiral_loss, n_ligand_chiral, batch_c6d_loss, batch_lddt,
                        batch_plddt, batch_plddt_loss
                    ])
                    exit(-1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 定义需要同步的损失变量列表
                loss_vars = [
                        batch_total_loss, batch_sm_token_loss, batch_bond_loss, batch_protein_sm_fape_loss,
                        batch_sm_protein_fape_loss, batch_protein_trans_loss, batch_protein_rots_loss,
                        batch_tors_loss, batch_protein_blen_loss, batch_protein_bang_loss, batch_ligand_coords_loss,
                        batch_protein_ligand_contact_loss, batch_ligand_bond_loss, batch_ligand_rigid_loss,
                        batch_ligand_chiral_loss, n_ligand_chiral, batch_c6d_loss, batch_lddt,
                        batch_plddt, batch_plddt_loss
                    ]
                # 合并所有标量后只进行一次跨进程同步，减少通信调用次数
                reduced_loss_vars = torch.stack(loss_vars)
                dist.all_reduce(reduced_loss_vars, op=dist.ReduceOp.SUM)
                batch_total_loss, batch_sm_token_loss, batch_bond_loss, batch_protein_sm_fape_loss, \
                    batch_sm_protein_fape_loss, batch_protein_trans_loss, batch_protein_rots_loss, \
                    batch_tors_loss, batch_protein_blen_loss, batch_protein_bang_loss, batch_ligand_coords_loss, \
                    batch_protein_ligand_contact_loss, batch_ligand_bond_loss, batch_ligand_rigid_loss, \
                    batch_ligand_chiral_loss, n_ligand_chiral, batch_c6d_loss, batch_lddt, \
                    batch_plddt, batch_plddt_loss = reduced_loss_vars.unbind()

                # 主进程记录日志
                if local_rank == 0:
                    writer.add_scalar("train/grad_norm", grad_norm.item(), log_steps)
                    writer.add_scalar("train/total_loss", batch_total_loss.item(), log_steps)
                    writer.add_scalar("train/c6d_loss", batch_c6d_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_protein/trans_loss", batch_protein_trans_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_protein/rots_loss", batch_protein_rots_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_protein/tors_loss", batch_tors_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_ligand/ligand_coords_loss", batch_ligand_coords_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_ligand/protein_ligand_contact_loss", batch_protein_ligand_contact_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_ligand/ligand_seq_loss", batch_sm_token_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_ligand/ligand_bond_token_loss", batch_bond_loss.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_lddt/lddt", batch_lddt.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_lddt/plddt", batch_plddt.item() / config.train.batch_size, log_steps)
                    writer.add_scalar("train_lddt/plddt_loss", batch_plddt_loss.item() / config.train.batch_size, log_steps)
                    if USE_GEOMETRY:
                        writer.add_scalar("train_protein/blen_loss", batch_protein_blen_loss.item() / config.train.batch_size, log_steps)
                        writer.add_scalar("train_protein/bang_loss", batch_protein_bang_loss.item() / config.train.batch_size, log_steps)
                        writer.add_scalar("train_ligand/ligand_bond_loss", batch_ligand_bond_loss.item() / config.train.batch_size, log_steps)
                        writer.add_scalar("train_ligand/ligand_rigid_loss", batch_ligand_rigid_loss.item() / config.train.batch_size, log_steps)
                        writer.add_scalar("train_ligand/ligand_chiral_loss", batch_ligand_chiral_loss.item() / n_ligand_chiral.item(), log_steps)
                log_steps += 1

                batch_total_loss = torch.tensor(0., device=local_rank)
                batch_sm_token_loss = torch.tensor(0., device=local_rank)
                batch_bond_loss = torch.tensor(0., device=local_rank)
                batch_protein_sm_fape_loss = torch.tensor(0., device=local_rank)
                batch_sm_protein_fape_loss = torch.tensor(0., device=local_rank)
                batch_protein_trans_loss = torch.tensor(0., device=local_rank)
                batch_protein_rots_loss = torch.tensor(0., device=local_rank)
                batch_tors_loss = torch.tensor(0., device=local_rank)
                batch_protein_blen_loss = torch.tensor(0., device=local_rank)
                batch_protein_bang_loss = torch.tensor(0., device=local_rank)
                batch_ligand_coords_loss = torch.tensor(0., device=local_rank)
                batch_protein_ligand_contact_loss = torch.tensor(0., device=local_rank)
                batch_ligand_bond_loss = torch.tensor(0., device=local_rank)
                batch_ligand_rigid_loss = torch.tensor(0., device=local_rank)
                batch_ligand_chiral_loss = torch.tensor(0., device=local_rank)
                n_ligand_chiral = torch.tensor(0., device=local_rank)
                batch_c6d_loss = torch.tensor(0., device=local_rank)
                batch_lddt = torch.tensor(0., device=local_rank)
                batch_plddt = torch.tensor(0., device=local_rank)
                batch_plddt_loss = torch.tensor(0., device=local_rank)

                batch_size += 1
                used_count = 0

                """
                当剩余的数据数量不足一个伪批次大小，并且经过了一轮优化后时直接跳过当前epoch；
                """
                if (i + 1) + config.train.pseudo_batch_size > data_total:
                    done_flag.fill_(1)
                # 所有进程同步 done_flag
                dist.all_reduce(done_flag, op=dist.ReduceOp.MAX)
                if done_flag.item() == 1:
                    break

            # 主进程更新进度条
            if local_rank == 0:
                pbar.update(1)

        # 可能存在部分进程反向传播了一些数据，因此这里需要清除梯度信息
        optimizer.zero_grad()
        # 保存断点
        if local_rank == 0:
            save_dict = {
                "last_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                "epoch": e,
                'log_name': log_name,
                'log_steps': log_steps,
                'random_state': random.getstate(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            }
            torch.save(save_dict, './save_model/checkpoint.ckpt')
            pbar.clear()
            pbar.close()
        # 等待所有进程完成epoch
        dist.barrier()


if __name__ == '__main__':
    # 输入尺寸会变，因此设置为False
    torch.backends.cudnn.benchmark = False
    # 固定cuda的随机数种子，每次返回的卷积算法将是确定的
    torch.backends.cudnn.deterministic = True

    # 启动训练函数
    train(LOCAL_RANK)
    # 清理分布式进程
    dist.destroy_process_group()
