import pickle
from utils import get_response, uniprot_structure_url

if __name__ == '__main__':
    uniprot_structures_dict = pickle.load(open('./uniprot_structures_dict.pkl', 'rb'))  # pdbbind+上的holo结构所属的uniprot实例中的引用结构数据

    new_uniprot_id = 'A0AA82WPE8'

    result = get_response(uniprot_structure_url % new_uniprot_id)
    if result.status_code == 200:
        uniprot_structures_dict[new_uniprot_id] = result.json()['results'][0]['uniProtKBCrossReferences']
        pickle.dump(uniprot_structures_dict, open('./uniprot_structures_dict.pkl', 'wb'))
