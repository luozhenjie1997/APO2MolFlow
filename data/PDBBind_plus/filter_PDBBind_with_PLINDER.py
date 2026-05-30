import os
import pickle
import copy
import gzip
import re
import warnings
import subprocess
import tempfile
import torch
import pandas as pd
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from Bio import SeqIO
from Bio.PDB import Structure, Model, Superimposer, PDBIO
from Bio import BiopythonWarning
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from process_apo import process_apo
from process_holo import process_holo
from model.utils.utils import parse_mol
from utils import load_config, error_holo_id, mol2_subst_names, get_ligand_coords_token, get_protein_chains, get_response, \
    rcsb_fasta_download_utl, align_seq, get_rcsb_pdb_and_assm_file, get_pdb_structure, get_cif_label_auth_mapping

"""
筛选出holo结构对应的apo结构。主要根据PLINDER（针对单链PLI系统）以及手动筛选（针对多链PLI系统）
PLINDER的结果截止至2024-4-30

以CIF形式存储的生物组装体没有MODEL形式，会以chain-*的形式表示

在PLINDER的关键字段：
system_id：系统id
system_ligand_chains：系统的配体所处的链，例如['1.B']。经rcsb重新命名
system_ligand_has_artifact：系统是否存在伪造配体，有则为True
system_ligand_has_oligo：系统是否存在寡聚配体，有则为True
ligand_ccd_code：配体的CCD代码

system_id的构成：
[PDB_ID]__[Biounit_ID]__[Protein_Chains]__[Ligand_Chains]。例如1abc__1__1.A_1.B__1.C
1abc：来自于 PDB 结构 1abc。
1：提取自它的第1号生物组装体。
1.A_1.B：这是一个界面结合口袋，由蛋白质的 A 链和 B 链共同提供相互作用残基。
1.C：结合在这个界面口袋里的配体是 C 链。
"""

# 忽略读取sdf时的警告
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=BiopythonWarning)

RESOLUTION = 3.  # 允许的最低分辨率的阈值
IDENTITY = 0.9  # apo序列和holo序列的一致性阈值（0-1）
POCKET_IDENTITY = 0.95  # apo口袋序列和holo口袋序列的一致性阈值（0-1）
LENGTH_COVERAGE = 0.9  # apo序列和holo序列的覆盖度阈值（0-1）
MIN_RMSD = 0.5  # apo和holo的最小rmsd
MIN_POCKET_RMSD = 1.2  # apo和holo的口袋区域的最小rmsd

mm_name1_patthen = re.compile('Name of Structure_1:\s*(\S+).*[\r\n]+Name of Structure_2:\s*(\S+)')
rmsd_pattern = re.compile('RMSD.*\d*\.\d*')
number_pattern = re.compile('\d*\.\d*')

config, config_name = load_config('../../config/base_config.yaml')
# PDBBind中具有生物作用的PL数据索引
index_df = pd.read_csv(config['dataset']['base_path'] + '/INDEX_general_PL_data.2024', usecols=[0, 1],
                       skiprows=5, sep='  ', names=['PDB code', 'resolution'])
uniprot_index_df = pd.read_csv(config['dataset']['base_path'] + '/INDEX_general_PL_name.2024',
                               skiprows=6, sep='  ', names=['PDB code', 'release year', 'Uniprot ID', 'protein name'])
# Q-BioLip中具有生物作用的PL数据索引
q_BioLip_relevant_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/Q-BioLiP_relevant.csv')
# Q-BioLip中具有生物作用的蛋白质-多肽数据
q_BioLip_protein_peptide_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/PIII.list',
                                          names=['PDB ID', 'Ligand ID', 'Ligand Name', 'Sites'])
# Q-BioLip中具有生物作用的蛋白质-DNA数据
q_BioLip_protein_dna_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/PDNA.list',
                                      names=['PDB ID', 'Ligand ID', 'Ligand Name', 'Sites'])
# Q-BioLip中具有生物作用的蛋白质-RNA数据
q_BioLip_protein_rna_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/PRNA.list',
                                      names=['PDB ID', 'Ligand ID', 'Ligand Name', 'Sites'])
q_BioLip_df_list = [q_BioLip_relevant_df, q_BioLip_protein_peptide_df, q_BioLip_protein_dna_df, q_BioLip_protein_rna_df]
# Q-BioLip标记的伪影配体列表
additives_df = pd.read_csv(config['dataset']['base_path'] + '/additives.tsv', sep='\t')
pdbbind_uniprot_structure_dict = pickle.load(open('./uniprot_structures_dict.pkl', 'rb'))  # pdbbind+上的holo结构所属的uniprot实例中的引用结构数据


def get_ca_atoms(chain):
    """
    提取一条链中所有标准氨基酸的 C-alpha (CA) 原子列表
    """
    ca_atoms = []
    for residue in chain:
        # hetero_flag 为 " " 表示这是一个标准的氨基酸残基（过滤掉水分子和配体杂质）
        if residue.id[0] == " " and residue.has_id("CA"):
            ca_atoms.append(residue["CA"])
    return ca_atoms


def get_apo(system_holo_seq_records, uniprot_apo_structure_reference):
    candidate_apo_id = uniprot_apo_structure_reference['id'].upper()
    q_BioLip_list = [df[df['PDB ID'].str.contains(candidate_apo_id.lower())] for df in q_BioLip_df_list]

    if sum([len(df) for df in q_BioLip_list[1:]]) > 0:  # 先排除掉含有非小分子配体的结构
        return None, False, None
    if len(q_BioLip_list[0]) > 0:
        candidate_apo_ligands = list(set(list(q_BioLip_list[0]['Ligand ID'].values)))
        if sum(additives_df['ligand'].isin(candidate_apo_ligands)) != len(candidate_apo_ligands):  # 排除掉含有非伪影配体的结构
            return None, False, None

    resolution = uniprot_apo_structure_reference['properties'][1]['value'].split(' ')[0]
    if resolution != '-' and float(resolution) <= RESOLUTION:  # 分辨率大于一定程度的结构才保留
        # 获取fasta文件，如果本地不存在则前往rcsb上下载
        if not os.path.exists(config['dataset']['apo_path'] + '/candidate/fasta/%s.fasta' % candidate_apo_id):
            response = get_response(rcsb_fasta_download_utl + candidate_apo_id)
            if response.status_code != 200:
                return None, False, resolution
            with open(config['dataset']['apo_path'] + '/candidate/fasta/%s.fasta' % candidate_apo_id, 'w') as f:
                f.write(response.text)
        # 为简化问题，apo结构只保留单体和同源寡聚体，排除掉异源寡聚体的数据
        try:
            candidate_apo_records = list(SeqIO.parse(config['dataset']['apo_path'] + '/candidate/fasta/%s.fasta' % candidate_apo_id, 'fasta'))
            if len(candidate_apo_records) > 1:
                _, _, identity_percent, _ = align_seq(str(candidate_apo_records[0].seq), str(candidate_apo_records[1].seq))
                if identity_percent < 1.:  # 避免某些情况下为记录了多个record的同源寡聚体而被排除
                    return None, False, resolution
        except:
            return None, False, resolution

        # 获取包含组装体的结构文件列表
        candidate_apo_id_list = get_rcsb_pdb_and_assm_file(config['dataset']['apo_path'] + '/candidate/pdb/%s.cif' % candidate_apo_id)

        # 获取潜在apo结构真正的链的数量
        candidate_apo_chains = [get_protein_chains(config['dataset']['apo_path'] + '/candidate/pdb/%s' % candidate_apo_id)
                                for candidate_apo_id in candidate_apo_id_list]

        """如果候选apo结构的亚基数量小于holo的亚基数量，则表明该候选apo结构无法分割，故直接排除"""
        for candidate_apo_id, candidate_apo_chain in zip(copy.deepcopy(candidate_apo_id_list), candidate_apo_chains):
            if candidate_apo_chain is None or len(candidate_apo_chain) < len(system_holo_seq_records):
                candidate_apo_id_list.remove(candidate_apo_id)
        if len(candidate_apo_id_list) == 0:
            return None, False, resolution

        """仅保留序列同源性大于指定阈值以及二者的结构为同等级大小（覆盖度大于指定阈值）的结构。对于同源寡聚体，其所有亚基的序列都是一致的。"""
        try:
            candidate_apo_seq = str(candidate_apo_records[0].seq)
            holo_seq = str(system_holo_seq_records[0].seq)
            _, _, identity_percent, coverage_pct = align_seq(candidate_apo_seq, holo_seq)
            if identity_percent < IDENTITY or coverage_pct < LENGTH_COVERAGE:
                return None, False, resolution
        except:
            return None, False, resolution

        return candidate_apo_id_list, True, resolution
    else:
        return None, False, resolution


def filter_apo(apo_info, holo_res_seqs, holo_pocket_atoms, holo_pocket_idx_mapping, holo_pocket_seqs, holo_chain_num):
    best_rmsd = float('inf')
    best_resolution = None
    best_apo = None
    best_apo_st = None
    best_apo_chains = None
    best_chain_mapping = None  # {holo_chain: apo_chain}

    # 初始化叠合计算器
    sup = Superimposer()

    for apo in apo_info[0]:  # 需要遍历每一个生物组状体，避免对称性问题
        apo_path = config['dataset']['apo_path'] + '/candidate/pdb/%s' % apo
        if '.gz' in apo:
            with gzip.open(apo_path, 'rt') as handle:
                apo_records = list(SeqIO.parse(handle, 'cif-seqres'))
            # 将压缩的cif文件写入临时文件
            with tempfile.NamedTemporaryFile(mode='w+t') as temp:
                with gzip.open(apo_path, 'rt') as gz:
                    temp.write(gz.read())
                result = subprocess.run(['./MMalign', temp.name, './holo3.pdb'], capture_output=True, text=True)
                rcsb_apo_st = get_pdb_structure(temp.name, auth_chains=False)
                apo_label_auth_mapping = get_cif_label_auth_mapping(temp.name)  # {auth_id: label_id}
        else:
            result = subprocess.run(['./MMalign', apo_path, './holo3.pdb'], capture_output=True, text=True)
            rcsb_apo_st = get_pdb_structure(apo_path, auth_chains=False)
            apo_records = list(SeqIO.parse(apo_path, 'cif-seqres'))
            apo_label_auth_mapping = get_cif_label_auth_mapping(apo_path)  # {auth_id: label_id}
        name = mm_name1_patthen.findall(result.stdout)[0]
        """获取最匹配的apo链id。对于MMalign，会将匹配成功的链id放到前面并一一对应"""
        match_holo_chains = name[1].split('/')[-1].split(':')[1:][:holo_chain_num]
        match_apo_chains = name[0].split('/')[-1].split(':')[1:][:holo_chain_num]
        match_apo_chains = [apo_label_auth_mapping[chain] for chain in match_apo_chains]  # 映射回label_id
        rmsd = float(number_pattern.findall(rmsd_pattern.findall(result.stdout)[0])[0])

        if '' in match_apo_chains or '' in match_holo_chains:  # 有无法匹配的链时不采用该结构
            continue

        select_apo_records = [record for record in apo_records if record.annotations['chain'] in match_apo_chains]
        apo_protein_seqs = [str(record.seq) for record in select_apo_records]
        pocket_alignment, _, _, pocket_coverage_pct = align_seq(''.join(holo_pocket_seqs.values()), ''.join(apo_protein_seqs))  # 计算apo和holo口袋序列的覆盖率
        # 当apo和holo的口袋的覆盖度达到一定阈值后才进行后续操作
        if pocket_coverage_pct >= POCKET_IDENTITY and rmsd < best_rmsd:
            best_resolution = apo_info[-1]
            best_rmsd = rmsd
            best_apo_st = rcsb_apo_st
            best_apo_chains = match_apo_chains
            best_apo = apo
            best_chain_mapping = {holo_chain: apo_chain for holo_chain, apo_chain in zip(name[1].split('/')[-1].split(':')[1:][:holo_chain_num],
                                                                                         match_apo_chains)}

    if best_apo_chains is not None:
        # 将选取的apo结构保存为一个新的Structure
        select_apo_protein_st = Structure.Structure(best_apo[:4])
        protein_model = Model.Model(0)
        for chain in best_apo_st[0]:
            if chain.id in best_apo_chains:
                chain_copy = copy.deepcopy(chain)  # 深拷贝，切断它与原结构的父子连结
                protein_model.add(chain_copy)  # 把它添加进新的 Model
        select_apo_protein_st.add(protein_model)  # 将新 Model 加入新 Structure

        # 获取apo结构实际解析的残基序列
        apo_res_seq = defaultdict(str)
        for apo_chain in select_apo_protein_st.get_chains():
            for res in apo_chain:
                if is_aa(res, standard=True) and res.id[0] == ' ' and 'CA' in res:
                    apo_res_seq[apo_chain.id] += seq1(res.get_resname())

        apo_atoms = {}
        for apo_chain in select_apo_protein_st.get_chains():
            apo_atoms[apo_chain.id] = get_ca_atoms(apo_chain)

        # 获取holo口袋残基到apo的索引映射并计算rmsd
        use_holo_pocket_atoms = []
        use_apo_atoms = []
        for pocket_chain, pocket_seq in holo_pocket_seqs.items():
            # 先获取apo和holo残基序列的索引映射
            apo_holo_alignment = align_seq(apo_res_seq[best_chain_mapping[pocket_chain]], holo_res_seqs[pocket_chain], return_only_alignment=True)
            aligned_positions = []
            for (start1, end1), (start2, end2) in zip(apo_holo_alignment.aligned[0], apo_holo_alignment.aligned[1]):  # 将区块解包并转换为1对1的坐标点
                for i in range(end1 - start1):
                    aligned_positions.append((start2 + i, start1 + i))  # (holo_idx, apo_idx)
            # 将holo的口袋残基索引映射到apo的索引
            apo_pocket_mapping = []
            for pocket_mapping in holo_pocket_idx_mapping[pocket_chain]:
                for mapping in aligned_positions:
                    if mapping[0] == pocket_mapping[0]:
                        apo_pocket_mapping.append((pocket_mapping[1], mapping[1]))
                        break
            for p1, p2 in apo_pocket_mapping:
                use_holo_pocket_atoms.append(holo_pocket_atoms[pocket_chain][p1])
                use_apo_atoms.append(apo_atoms[best_chain_mapping[pocket_chain]][p2])

        if len(use_apo_atoms) == 0:
            return None
        sup.set_atoms(use_holo_pocket_atoms, use_apo_atoms)  # 执行结构叠合，获取apo和holo口袋区域的rmsd


    return {'select_apo_name': best_apo, 'select_apo_chains': best_apo_chains, 'select_apo_resolution': best_resolution,
            'select_apo_rmsd': best_rmsd, 'select_apo_pocket_rmsd': sup.rms}


if __name__ == '__main__':
    no_use_holo_ids = []
    use_holo_ids = []
    manual_apo_holo_list = []
    error_protein_list = []
    finish_num = 0

    pbar = tqdm(range(len(index_df)))

    for p_i, (_, line) in zip(pbar, index_df.iterrows()):
        pbar.set_description(f"成功处理{finish_num}个结构")

        holo_id = line['PDB code']
        resolution = line['resolution']

        holo_id = '4k7o'  # 生物组装体，只有一种对称情况
        # holo_id = '1b8n'  # 生物组装体，有多种对称情况
        # holo_id = '8ao3'  # 含有共价结合信息
        # resolution = 3.
        # if holo_id != '1tqf':
        #     continue
        # if p_i < 3120:
        #     continue
        # if p_i > 30000 or p_i < 20000:
        #     continue

        if holo_id in error_holo_id:
            no_use_holo_ids.append(holo_id)
            continue

        if resolution == 'NMR' or float(resolution) > RESOLUTION:  # 跳过NMR解析的结果以及解析分辨率低于指定阈值的结构
            no_use_holo_ids.append(holo_id)
            continue

        q_BioLip_list = [df[df['PDB ID'].str.contains(holo_id.lower())] for df in q_BioLip_df_list[1:]]
        if sum([len(df) for df in q_BioLip_list]) > 0:  # 排除掉含有非小分子配体的结构
            no_use_holo_ids.append(holo_id)
            continue

        uniprot_id = uniprot_index_df[uniprot_index_df['PDB code'] == holo_id]['Uniprot ID'].values[0]  # 获取uniprot id
        try:
            uniprot_structure_references = pdbbind_uniprot_structure_dict[uniprot_id]  # 获取所有rcsb结构交叉引用的信息
        except:
            manual_apo_holo_list.append(holo_id)  # 可能是因为原来的uniprot被拆分了或者其他原因，导致该uniprot不存在
            continue

        if len(uniprot_structure_references) == 0:  # 该uniprot记录下没有对应的实验结构
            manual_apo_holo_list.append(holo_id)
            continue

        # 获取配体名称列表
        ligand_names = list(set(mol2_subst_names(config['dataset']['holo_path'] + '/%s/%s/%s_ligand.mol2' % (holo_id[1:3], holo_id, holo_id))))

        holo_protein_chains, holo_protein_st = get_protein_chains(config['dataset']['holo_path'] + '/%s/%s/%s_protein.pdb' % (holo_id[1:3], holo_id, holo_id), return_st=True)
        holo_pocket_chains, holo_pocket_st = get_protein_chains(config['dataset']['holo_path'] + '/%s/%s/%s_pocket.pdb' % (holo_id[1:3], holo_id, holo_id), return_st=True)

        # 将选区的蛋白质chain对象合并为一个structure
        select_holo_protein_st = Structure.Structure(holo_id)
        protein_model = Model.Model(0)
        for chain in holo_pocket_chains:
            chain_copy = copy.deepcopy(holo_protein_st[0][chain])  # 深拷贝，切断它与原结构的父子连结
            protein_model.add(chain_copy)  # 把它添加进新的 Model
        select_holo_protein_st.add(protein_model)  # 将新 Model 加入新 Structure

        # 将选取的holo结构保存为cif文件
        io = PDBIO()
        io.set_structure(select_holo_protein_st)
        io.save("holo3.pdb")

        # 获取holo结构的完整序列
        holo_protein_records = list(SeqIO.parse(config['dataset']['holo_path'] + '/%s/%s/%s_protein.pdb' % (holo_id[1:3], holo_id, holo_id), 'pdb-seqres'))
        holo_protein_records = [record for record in holo_protein_records if record.annotations['chain'] in holo_pocket_chains]
        holo_protein_seqs = [str(record.seq) for record in holo_protein_records]
        holo_res_records = list(SeqIO.parse(config['dataset']['holo_path'] + '/%s/%s/%s_protein.pdb' % (holo_id[1:3], holo_id, holo_id), 'pdb-atom'))
        holo_res_seqs = {record.annotations['chain']: str(record.seq) for record in holo_res_records if
                         record.annotations['chain'] in holo_pocket_chains}

        initial_apo_holo_list = []
        if len(uniprot_structure_references) < 50:  # 只有数据量大于一定值时，多进程才会比单进程迭代快
            for uniprot_apo_structure_reference in uniprot_structure_references:
                candidate_apo_id, is_apo, resolution = get_apo(holo_protein_records, uniprot_apo_structure_reference)
                if is_apo:
                    initial_apo_holo_list.append((candidate_apo_id, holo_id, resolution))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=52) as executor:  # max_workers根据实际情况设置，不设置max_workers则默认使用全部核心
                futures = []
                for uniprot_apo_structure_reference in uniprot_structure_references:
                    futures.append(executor.submit(get_apo, holo_protein_records, uniprot_apo_structure_reference))

                # 多进程并行执行
                for future in futures:
                    candidate_apo_id, is_apo, resolution = future.result()
                    if is_apo:
                        initial_apo_holo_list.append((candidate_apo_id, holo_id, resolution))

        if len(initial_apo_holo_list) == 0:
            manual_apo_holo_list.append(holo_id)
            continue

        initial_apo_holo_list = sorted(initial_apo_holo_list, key=lambda x: x[2])  # 按照解析分辨率排序

        holo_chain_num = len(holo_pocket_chains)

        # 获取holo口袋区域的CA原子
        holo_pocket_atoms = defaultdict(list)
        for holo_chain in holo_pocket_st.get_chains():
            atoms = get_ca_atoms(holo_chain)
            if len(atoms) > 0:
                holo_pocket_atoms[holo_chain.id] = atoms

        # 将holo结构的口袋区域和残基序列的索引对应起来
        holo_pocket_idx_mapping = defaultdict(list)
        for holo_chain in select_holo_protein_st.get_chains():
            pocket_atoms = holo_pocket_atoms[holo_chain.id]
            pocket_idx = 0
            mapping = []
            for i, res in enumerate(holo_chain):
                if pocket_atoms[pocket_idx].parent.id == res.id:
                    mapping.append((i, pocket_idx))
                    pocket_idx += 1
                if pocket_idx == len(pocket_atoms):  # 所有口袋残基遍历完毕，则无需继续遍历
                    break
            holo_pocket_idx_mapping[holo_chain.id] = mapping

        holo_pocket_records = list(SeqIO.parse(config['dataset']['holo_path'] + '/%s/%s/%s_pocket.pdb' % (holo_id[1:3], holo_id, holo_id), 'pdb-atom'))
        holo_pocket_seqs = {record.annotations['chain']: str(record.seq).replace('X', '') for record in holo_pocket_records}

        apo_info_list = []
        is_has_error_protein = False
        if len(initial_apo_holo_list) == 1:
            for apo_info in initial_apo_holo_list:
                result = filter_apo(apo_info, holo_res_seqs, holo_pocket_atoms, holo_pocket_idx_mapping, holo_pocket_seqs, holo_chain_num)
                if result is None:
                    is_has_error_protein = True
                elif result['select_apo_name'] is not None:
                    apo_info_list.append(result)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=52) as executor:  # max_workers根据实际情况设置，不设置max_workers则默认使用全部核心
                futures = []
                for apo_info in initial_apo_holo_list:
                    futures.append(executor.submit(filter_apo, apo_info, holo_res_seqs, holo_pocket_atoms, holo_pocket_idx_mapping,
                                                   holo_pocket_seqs, holo_chain_num))

                # 多进程并行执行
                for future in futures:
                    result = future.result()
                    if result is None:
                        is_has_error_protein = True
                    elif result['select_apo_name'] is not None:
                        apo_info_list.append(result)

        if is_has_error_protein:
            error_protein_list.append(holo_id)
            continue
        if len(apo_info_list) == 0:
            manual_apo_holo_list.append(holo_id)
            continue

        apo_info_list = sorted(apo_info_list, key=lambda x: x['select_apo_resolution'])  # 按照分辨率排序
        # select_apo_info = apo_info_list[0]
        """直接选取分辨率最高的那个，并确保口袋rmsd要大于最小值，即追求最清晰的物理变化。若没有，则退化至直接选取分辨率最高的那个"""
        select_apo_info = None
        for apo_info in apo_info_list:
            if apo_info['select_apo_pocket_rmsd'] > MIN_POCKET_RMSD:
                select_apo_info = apo_info
        if select_apo_info is None:
            select_apo_info = apo_info_list[0]

        # 选取对应的apo部分
        rcsb_apo_st = get_pdb_structure(config['dataset']['apo_path'] + '/candidate/pdb/%s' % select_apo_info['select_apo_name'], auth_chains=False)
        select_apo_protein_st = Structure.Structure(select_apo_info['select_apo_name'][:4])
        protein_model = Model.Model(0)
        for chain in rcsb_apo_st[0]:
            if chain.id in select_apo_info['select_apo_chains']:
                chain_copy = copy.deepcopy(chain)
                protein_model.add(chain_copy)
        select_apo_protein_st.add(protein_model)
        if '.gz' in select_apo_info['select_apo_name']:
            with gzip.open(config['dataset']['apo_path'] + '/candidate/pdb/%s' % select_apo_info['select_apo_name'], 'rt') as handle:
                apo_seq_records = list(SeqIO.parse(handle, 'cif-seqres'))
        else:
            apo_seq_records = list(SeqIO.parse(config['dataset']['apo_path'] + '/candidate/pdb/%s' % select_apo_info['select_apo_name'], 'cif-seqres'))
        apo_seq_records = [record for record in apo_seq_records if record.annotations['chain'] in select_apo_info['select_apo_chains']]  # 获取apo对应的序列record

        result = {}
        for key, item in parse_mol(config['dataset']['holo_path'] + '/%s/%s/%s_ligand.sdf' % (holo_id[1:3], holo_id, holo_id)).items():
            result['ligand_' + key] = item
        if result is None:
            no_use_holo_ids.append(holo_id)
            continue
        template_coords, template_tokens = get_ligand_coords_token(Chem.MolFromMolFile(config['dataset']['holo_path'] + '/%s/%s/%s_ligand.sdf'
                                                                                       % (holo_id[1:3], holo_id, holo_id), sanitize=True, removeHs=True))
        if template_coords is None:
            no_use_holo_ids.append(holo_id)
            continue
        result['ligand_atom_coords'] = template_coords  # 替换为真实的配体坐标
        result['ligand_atom_mask'] = torch.ones(template_coords.shape[0]).bool()

        apo_result = process_apo(select_apo_protein_st, apo_seq_records)  # 预处理apo部分

        pocket_residues = []
        for chain in holo_pocket_st.get_chains():
            for res in chain:
                if res.resname != 'HOH':  # 不添加水分子
                    pocket_residues.append(res)
        holo_result = process_holo(select_holo_protein_st, holo_protein_records, pocket_residues, ligand_names, result)  # 预处理holo部分

        # 合并所有预处理结果
        del result['ligand_obmol']
        result.update(apo_result)
        result.update(holo_result)
        result['apo_holo_pocket_rmsd'] = select_apo_info['select_apo_pocket_rmsd']

        save_name = holo_id + '__%s|%s__%s' % ('_'.join(holo_pocket_chains), select_apo_info['select_apo_name'][:4], '_'.join(select_apo_info['select_apo_chains']))
        torch.save(result, config['dataset']['holo_path'] + '/%s/%s/%s.pt' % (holo_id[1:3], holo_id, save_name))

        use_holo_ids.append(save_name)
        finish_num += 1

    pickle.dump(use_holo_ids, open('./final_result/use_holo_ids.pkl', 'wb'))
    pickle.dump(no_use_holo_ids, open('./final_result/no_use_holo_ids.pkl', 'wb'))
    pickle.dump(manual_apo_holo_list, open('./final_result/manual_apo_holo_list.pkl', 'wb'))
    pickle.dump(error_protein_list, open('./final_result/error_protein_list.pkl', 'wb'))
