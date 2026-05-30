import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from utils import mol2_subst_names

MAX_MW = 1000.

if __name__ == '__main__':
    apo_holo_list = pickle.load(open('./final_result/use_holo_ids.pkl', 'rb'))
    alter_apo_holo_list = []

    pbar = tqdm(range(len(apo_holo_list)))

    for _, apo_holo_id in zip(pbar, apo_holo_list):
        holo_id = apo_holo_id[:4]
        ligand_path = '/root/autodl-tmp/dataset/APO2MolFlow/holo/%s/%s/%s_ligand.sdf' % (holo_id[1: 3], holo_id, holo_id)

        ligand_name = mol2_subst_names(ligand_path.replace('.sdf', '.mol2'))[0]  # 获取配体名称

        mol = Chem.SDMolSupplier(ligand_path)[0]

        try:
            mw_avg = Descriptors.MolWt(mol)  # 平均分子重量
            # mw_exact = Descriptors.ExactMolWt(mol)  # 精确质量（monoisotopic mass）
        except:
            continue

        if mw_avg <= MAX_MW:
            alter_apo_holo_list.append(apo_holo_id)

    pickle.dump(alter_apo_holo_list, open('./final_result/use_holo_ids_filter_mw.pkl', 'wb'))
