import pickle
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from utils import get_response, uniprot_structure_url, load_config

if __name__ == '__main__':
    config, config_name = load_config('../../config/base_config.yaml')

    try:
        uniprot_structures_dict = pickle.load(open('./uniprot_structures_dict1.pkl', 'rb'))
    except:
        uniprot_structures_dict = {}
    find_keys = list(uniprot_structures_dict.keys())

    # PDBBind中具有生物作用的PL数据索引
    index_df = pd.read_csv(config['dataset']['base_path'] + '/INDEX_general_PL_name.2024',
                           skiprows=6, sep='  ', names=['PDB code', 'release year', 'Uniprot ID', 'protein name'])

    plinder_df = pq.ParquetFile('/root/autodl-tmp/dataset/PLINDER/2024-06_v2_index_annotation_table.parquet').read().to_pandas()

    pbar = tqdm(range(len(index_df)))

    for _, (_, line) in zip(pbar, index_df.iterrows()):
        uniprot_id = line['Uniprot ID']
        if uniprot_id in find_keys:
            continue

        if '-' not in uniprot_id:
            uniprot_response = get_response(uniprot_structure_url % uniprot_id)
            if uniprot_response.status_code == 200:
                try:
                    uniprot_structures_dict[uniprot_id] = uniprot_response.json()['results'][0]['uniProtKBCrossReferences']
                except:
                    print(uniprot_id)
                    continue
    pickle.dump(uniprot_structures_dict, open('./uniprot_structures_dict1.pkl', 'wb'))