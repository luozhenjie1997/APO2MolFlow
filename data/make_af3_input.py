import json
import pickle
import pandas as pd
from tqdm import tqdm
from Bio.PDB import parse_pdb_header
from Bio import SeqIO
from PDBBind_plus.utils import load_config, align_seq

"""
对于实在无法找到实验apo结构的holo结构，则使用af3预测apo结构。该代码用于准备af3所需的输入json
"""

RESOLUTION = 4.  # 允许的最低分辨率的阈值
MAX_CHAIN_NUM = 2
config, config_name = load_config('../config/base_config.yaml')

# Q-BioLip中具有生物作用的PL数据索引
q_BioLip_relevant_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/Q-BioLiP_relevant.csv')
# PDBBind中具有生物作用的PL数据索引
index_df = pd.read_csv(config['dataset']['base_path'] + '/INDEX_general_PL_data.2024',
                       skiprows=5, sep='  ', usecols=[0, 1], names=['PDB code', 'resolution'])
q_BioLip_protein_peptide_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/PIII.list',
                                          names=['PDB ID', 'Ligand ID', 'Ligand Name', 'Sites'])
# Q-BioLip中具有生物作用的蛋白质-DNA数据
q_BioLip_protein_dna_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/PDNA.list',
                                      names=['PDB ID', 'Ligand ID', 'Ligand Name', 'Sites'])
# Q-BioLip中具有生物作用的蛋白质-RNA数据
q_BioLip_protein_rna_df = pd.read_csv(config['dataset']['Q_BioLip_path'] + '/PRNA.list',
                                      names=['PDB ID', 'Ligand ID', 'Ligand Name', 'Sites'])
q_BioLip_df_list = [q_BioLip_relevant_df, q_BioLip_protein_peptide_df, q_BioLip_protein_dna_df, q_BioLip_protein_rna_df]

use_holo_list = pickle.load(open('/mnt/disk4/zyzhou/APO2MolFlow/use_holo_list.pkl', 'rb'))

if __name__ == '__main__':
    pbar = tqdm(range(len(index_df)))

    no_use_holo = []

    for i, (_, line) in zip(pbar, index_df.iterrows()):
        holo_pdb_id = line['PDB code']

        if holo_pdb_id in use_holo_list:  # 排除掉有真实apo结构的holo结构
            continue

        resolution = line['resolution']
        if resolution == 'NMR' or float(resolution) > RESOLUTION:  # 暂不使用NMR结构以及分辨率不够的结构
            continue

        # 排除含有非小分子且具有生物学作用的配体的holo结构
        holo_BioLip_len_list = [len(df[df['PDB ID'].str.contains(holo_pdb_id.lower())]) for df in q_BioLip_df_list[1:]]
        if sum(holo_BioLip_len_list) > 0:
            no_use_holo.append(holo_pdb_id)
            continue

        """为简化问题，holo结构只保留单体和同源寡聚体，排除掉异源寡聚体的数据"""
        try:
            records = list(SeqIO.parse(config['dataset']['holo_path'] + '/%s/%s/%s.fasta' % (holo_pdb_id[1:3], holo_pdb_id, holo_pdb_id), 'fasta'))
            if len(records) > 1:
                if len(records) <= MAX_CHAIN_NUM:
                    _, identity_percent, _ = align_seq(str(records[0].seq), str(records[1].seq))
                    if identity_percent < 1.:  # 避免某些情况下fasta记录了多个record，但实际上是同源寡聚体
                        no_use_holo.append(holo_pdb_id)
                        continue
                else:
                    no_use_holo.append(holo_pdb_id)
                    continue

            if len(records) > 1:
                holo_chains = []
                for record in records:
                    holo_chains.extend(
                        record.description.split('|')[1].replace('Chains ', '').replace('Chain ', '').split(', '))
            else:
                holo_chains = records[0].description.split('|')[1].replace('Chains ', '').replace('Chain ', '').split(', ')
        except:
            continue

        if len(holo_chains) > MAX_CHAIN_NUM:  # 只使用链数量不大于特定阈值的结构
            no_use_holo.append(holo_pdb_id)
            continue

        sequences = []
        for chain in holo_chains:
            sequences.append({'protein': {'id': chain.split('[')[0], 'sequence': str(records[0].seq)}})

        save_dict = {'name': holo_pdb_id, 'sequences': sequences, 'modelSeeds': [2026], "dialect": "alphafold3", "version": 1}
        json.dump(save_dict, open('/home/zyzhou/lzj/af_input/%s.json' % holo_pdb_id, 'w'), indent=4, separators=(', ', ': '))
