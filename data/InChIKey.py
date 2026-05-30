import csv
import os
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import inchi

INPUT_GLOB = "ligands/*.mol2"        # 修改为你的 mol2 路径模式
OUTPUT_TABLE = "ligand_inchikeys.csv"
DEDUP_REPORT = "duplicates.txt"

if __name__ == '__main__':
    # PL数据索引
    index_df = pd.read_csv('/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/INDEX_general_PL_name.2024',
                           skiprows=6, sep='  ', names=['PDB code', 'release year', 'Uniprot ID', 'protein name'])

    records = []
    groups = defaultdict(list)

    for _, line in index_df.iterrows():
        holo_pdb_id = list(line)[0]

        # PDBBind根据结构提交年份来区分数据
        if os.path.exists('/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/1981-2000/%s' % holo_pdb_id):
            holo_path = '/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/1981-2000'
        elif os.path.exists('/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/2001-2010/%s' % holo_pdb_id):
            holo_path = '/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/2001-2010'
        elif os.path.exists('/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/2011-2020/%s' % holo_pdb_id):
            holo_path = '/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/2011-2020'
        else:
            holo_path = '/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/2021-2023'

        mol = Chem.MolFromMol2File(holo_path + '/%s/%s_ligand.mol2' % (holo_pdb_id, holo_pdb_id), sanitize=True, removeHs=False)
        if mol is None:
            print(f"[WARN] 无法解析: {holo_path + '/%s/%s_ligand.mol2' % (holo_pdb_id, holo_pdb_id)}")
            continue
        key = inchi.MolToInchiKey(mol)
        records.append((key, holo_path + '/%s/%s_ligand.mol2' % (holo_pdb_id, holo_pdb_id)))
        groups[key].append(holo_path + '/%s/%s_ligand.mol2' % (holo_pdb_id, holo_pdb_id))

    with open(OUTPUT_TABLE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["InChIKey", "File"])
        writer.writerows(records)

    with open(DEDUP_REPORT, "w", encoding="utf-8") as f:
        for key, files in groups.items():
            if len(files) > 1:
                f.write(f"{key}\n")
                f.write("\n".join(f"  - {fp}" for fp in files))
                f.write("\n\n")

    print("InChIKey 表格写入:", OUTPUT_TABLE)
    print("重复记录写入:", DEDUP_REPORT)