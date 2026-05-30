import pickle
import os
import pandas as pd
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from utils import rcsb_pdb_download_url, rcsb_ligand_sdf_url, get_response, mol2_subst_names, load_config, get_protein_chains

config, config_name = load_config('../../config/base_config.yaml')
# PDBBind中具有生物作用的PL数据索引
index_df = pd.read_csv(config['dataset']['base_path'] + '/INDEX_general_PL_name.2024',
                       skiprows=6, sep='  ', names=['PDB code', 'release year', 'Uniprot ID', 'protein name'])
apo_holo_dict = pickle.load(open(config['dataset']['base_path'] + '/use_single_apo_holo_dict.pkl', 'rb'))

if __name__ == '__main__':
    pbar = tqdm(range(len(index_df)))

    for _, (_, line) in zip(pbar, index_df.iterrows()):
        holo_id = line['PDB code']
        # 获取配体名称
        # ligand_name_list = list(set(mol2_subst_names(config['dataset']['holo_path'] + '/%s/%s/%s_ligand.mol2' % (holo_id[1:3], holo_id, holo_id))))
        # for ligand_name in ligand_name_list:
        #     if not os.path.exists(config['dataset']['base_path'] + '/ligand_rcsb/%s.cif' % ligand_name):
        #         response = get_response(rcsb_ligand_sdf_url + ligand_name + '.cif')
        #         with open(config['dataset']['base_path'] + '/ligand_rcsb/%s.cif' % ligand_name, 'w') as f:
        #             f.write(response.text)
        #     if not os.path.exists(config['dataset']['base_path'] + '/ligand_rcsb/%s.sdf' % ligand_name):
        #         response = get_response(rcsb_ligand_sdf_url + ligand_name + '_ideal.sdf')
        #         with open(config['dataset']['base_path'] + '/ligand_rcsb/%s.sdf' % ligand_name, 'w') as f:
        #             f.write(response.text)

        # response = get_response(rcsb_pdb_download_url + holo_id + '.cif')
        # if response.status_code == 200:
        #     with open(config['dataset']['base_path'] + '/holo_rcsb/%s.cif' % holo_id, 'w') as f:
        #         f.write(response.text)
        # else:
        #     print("下载失败，请重试")
        #     exit(-1)

        # 确认是否为生物组装体，如果是的话则去下载完整的结构
        pdbbind_chains = get_protein_chains(config['dataset']['holo_path'] + '/%s/%s/%s_protein.pdb' % (holo_id[1:3], holo_id, holo_id))
        rcsb_chains = get_protein_chains(config['dataset']['base_path'] + '/holo_rcsb/%s.cif' % holo_id)
        header_dict = MMCIF2Dict(config['dataset']['base_path'] + '/holo_rcsb/%s.cif' % holo_id)
        assembly_gen_assembly_ids = list(set(header_dict['_pdbx_struct_assembly_gen.assembly_id']))
        assembly_oper_expression = list(set(header_dict['_pdbx_struct_assembly_gen.oper_expression']))
        if (len(header_dict['_pdbx_struct_assembly_gen.assembly_id']) > 1 and len(pdbbind_chains) > len(rcsb_chains)) or \
                (len(assembly_oper_expression[0].split(',')) > 1):
            for assembly_gen_assembly_id in assembly_gen_assembly_ids:
                holo_name = '%s-assembly%s.cif.gz' % (holo_id, assembly_gen_assembly_id)
                if not os.path.exists(config['dataset']['base_path'] + '/holo_rcsb/%s' % holo_name):
                    response = get_response(rcsb_pdb_download_url + holo_name)
                    with open(config['dataset']['base_path'] + '/holo_rcsb/%s' % holo_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):  # 使用流式下载
                            # 过滤掉保持连接的空块
                            if chunk:
                                f.write(chunk)
