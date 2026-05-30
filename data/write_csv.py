import pickle
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import parse_pdb_header
from PDBBind_plus.utils import read_seq_from_pdb, get_amino_acid_coords, get_protein_chains

if __name__ == '__main__':
    apo_holo_with_cluster_dict = pickle.load(open('/data/user1/python/APO2MolFlow/data/PDBBind_plus/filter_apo_holo_with_cluster_list.pkl', 'rb'))

    df_dict = {'apo_id': [], 'apo_chain': [], 'holo_id': [], 'holo_chain': [], 'pocket_idx': []}

    pbar = tqdm(range(len(apo_holo_with_cluster_dict)))

    for _, (holo_id, apo_ids) in zip(pbar, apo_holo_with_cluster_dict.items()):
        try:
            holo_chain_list = get_protein_chains('/data/user1/data/APO2MolFlow/holo/%s/%s/%s_protein.pdb' % (holo_id[1:3], holo_id, holo_id), min_residues=1)
        except:
            holo_chain_list = get_protein_chains('/data/user1/data/APO2MolFlow/holo/%s/%s/%s_protein.cif' % (holo_id[1:3], holo_id, holo_id), min_residues=1)
        holo_records = list(SeqIO.parse('/data/user1/data/APO2MolFlow/holo/%s/%s/%s.fasta' % (holo_id[1:3], holo_id, holo_id), 'fasta'))

        # 获取口袋区域所在的链id和残基编号
        holo_pocket_idx_dict = {}
        holo_pocket_seqs = read_seq_from_pdb('/data/user1/data/APO2MolFlow/holo/%s/%s/%s_pocket.pdb' % (holo_id[1:3], holo_id, holo_id))
        for holo_pocket_chain in holo_pocket_seqs.keys():
            residue_idx_list, _, _, _, _ = get_amino_acid_coords('/data/user1/data/APO2MolFlow/holo/%s/%s/%s_pocket.pdb' % (holo_id[1:3], holo_id, holo_id),
                                                                 chain_id=holo_pocket_chain)
            holo_pocket_idx = []
            icode_num = 0
            for residue_idx in residue_idx_list:
                if residue_idx[1][2] != ' ':  # 包含插入码需要特殊处理
                    icode_num += 1
                holo_pocket_idx.append(residue_idx[1][1] + icode_num)
            holo_pocket_idx_dict[holo_pocket_chain] = holo_pocket_idx

        for apo_id in apo_ids:
            df_dict['apo_id'].append(apo_id)
            df_dict['holo_id'].append(holo_id)
            apo_records = list(SeqIO.parse('/data/user1/data/APO2MolFlow/apo/candidate_apo_fasta/%s.fasta' % apo_id, 'fasta'))
            apo_seq_list = []
            try:
                header_dict = parse_pdb_header('/data/user1/data/APO2MolFlow/apo/candidate_apo_pdb/%s.pdb' % apo_id)
            except:
                header_dict = parse_pdb_header('/data/user1/data/APO2MolFlow/apo/candidate_apo_pdb/%s.cif' % apo_id)
            is_assembly = False
            # 确认是否为生物组装体
            for _, biomoltrans in header_dict['biomoltrans'].items():
                if len(biomoltrans) > 4:
                    is_assembly = True
                    break
            if is_assembly:
                try:
                    apo_chain_list = get_protein_chains('/data/user1/data/APO2MolFlow/apo/candidate_apo_pdb/%s.pdb1.gz' % apo_id)
                except:
       