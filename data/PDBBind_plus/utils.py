import Bio.PDB.Structure
import requests
import time
import re
import numpy as np
import mdtraj as md
import os
import gzip
import yaml
import torch
import subprocess
from easydict import EasyDict
from io import StringIO
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from Bio.PDB import is_aa
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from model.utils.chemical import aa2num

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",  # 模拟 Chrome 浏览器
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",  # 表示从谷歌跳转过来
}

uniprot_structure_url = 'https://rest.uniprot.org/uniprotkb/search?query=accession:%s&fields=structure_3d'
rcsb_pdb_download_url = 'https://files.rcsb.org/download/'
pdbbind_fasta_url = 'https://www.pdbbind-plus.org.cn:11033/api/browser/fasta/%s.txt'
rcsb_fasta_download_utl = 'https://www.rcsb.org/fasta/entry/'
rcsb_entry_url = 'https://data.rcsb.org/rest/v1/core/entry/'
rcsb_ligand_sdf_url = 'https://files.rcsb.org/ligands/download/'

mm_name1_patthen = re.compile('Name of Structure_1:\s*(\S+).*[\r\n]+Name of Structure_2:\s*(\S+)')
number_pattern = re.compile('\d*\.\d*')
rmsd_pattern = re.compile('RMSD.*\d*\.\d*')

# 此内容摘自BioLiP网站上名为“PDB中的金属”链接的查询字符串。该列表的离子是具有生物相互作用的
METAL_RES_NAMES = ['LA','NI','3CO','K','CR','ZN','CD','PD','TB','YT3','OS','EU','NA','RB','W','YB','HO3',
                   'CE','MN','TL','LI','MN3','AU3','AU','EU3','AL','3NI','FE2','PT','FE','CA','AG','CU1',
                   'LU','HG','CO','SR','MG','PB','CS','GA','BA','SM','SB','CU','MO','CU2']

one_letter = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X"]

one_letter_token = {aa: i for i, aa in enumerate(one_letter)}

error_holo_id = ['1lvk']

aa2long=[
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #0  ala
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), #1  arg
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), #2  asn
        (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #3  asp
        (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), #4  cys
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), #5  gln
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), #6  glu
        (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), #7  gly
        (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), #8  his
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), #9  ile
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), #10 leu
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), #11 lys
        (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), #12 met
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None), #13 phe
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), #14 pro
        (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), #15 ser
        (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), #16 thr
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), #17 trp
        (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None), #18 tyr
        (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), #19 val
        (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #20 unk
]

aa2long_noblank = [
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "3HB", None, None, None, None, None, None, None, None),  # 0ala
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HD", "2HD", "HE", "1HH1", "2HH1", "1HH2", "2HH2"), # 1arg
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HD2", "2HD2", None, None, None, None, None, None, None),  # 2asn
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", None, None, None, None, None, None, None, None, None),  # 3asp
    ("N", "CA", "C", "O", "CB", "SG", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "HG", None, None, None, None, None, None, None, None),  # 4cys
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HE2", "2HE2", None, None, None, None, None),  # 5gln
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HG", "2HG", None, None, None, None, None, None, None),  # 6glu
    ("N", "CA", "C", "O", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "1HA", "2HA", None, None, None, None, None, None, None, None, None, None),
    # 7gly
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "2HD", "1HE", "2HE", None, None, None, None, None, None),
    # 8his
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "HB", "1HG2", "2HG2", "3HG2", "1HG1", "2HG1", "1HD1", "2HD1", "3HD1", None, None),
    # 9ile
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "HG", "1HD1", "2HD1", "3HD1", "1HD2", "2HD2", "3HD2", None, None),
    # 10leu
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HD", "2HD", "1HE", "2HE", "1HZ", "2HZ", "3HZ"),
    # 11lys
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HE", "2HE", "3HE", None, None, None, None),
    # 12met
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HD", "2HD", "1HE", "2HE", "HZ", None, None, None, None),
    # 13phe
    ("N", "CA", "C", "O", "CB", "CG", "CD", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "HA", "1HB", "2HB", "1HG", "2HG", "1HD", "2HD", None, None, None, None, None, None),
    # 14pro
    ("N", "CA", "C", "O", "CB", "OG", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HG", "HA", "1HB", "2HB", None, None, None, None, None, None, None, None),
    # 15ser
    ("N", "CA", "C", "O", "CB", "OG1", "CG2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HG1", "HA", "HB", "1HG2", "2HG2", "3HG2", None, None, None, None, None, None),
    # 16thr
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2", None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HD", "1HE", "HZ2", "HH2", "HZ3", "HE3", None, None, None),
    # 17trp
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "1HD", "1HE", "2HE", "2HD", "HH", None, None, None, None),
    # 18tyr
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "HB", "1HG1", "2HG1", "3HG1", "1HG2", "2HG2", "3HG2", None, None, None, None),
    # 19val
    ("N", "CA", "C", "O", "CB", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "H", "HA", "1HB", "2HB", "3HB", None, None, None, None, None, None, None, None),
    # 20unk
]

notconsidered = ['O-O', 'F-F', 'H-H']  # 共价结合不需要考虑的结合原子

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name

def get_response(url):
    fail_flag = False
    for attempt in range(10):
        try:
            response = requests.get(url, headers=headers)
            fail_flag = False
            break
        except requests.exceptions.SSLError as e:  # SSL错误时重试
            fail_flag = True
            time.sleep(1)
        except requests.exceptions.ConnectionError as e:  # 连接出现异常时重试
            fail_flag = True
            time.sleep(1)
    if fail_flag:  # 多次重试失败，直接退出
        exit(-1)
    return response

def get_pdb_structure(pdb_file, auth_chains=True):
    if '.pdb' in pdb_file:
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True, auth_chains=auth_chains)

    if '.gz' in pdb_file:
        with gzip.open(pdb_file, 'rt') as handle:
            structure = parser.get_structure('protein', handle)
    else:
        structure = parser.get_structure('protein', pdb_file)

    return structure

def get_protein_chains(structure, min_residues=1, model_id=None, auth_chains=True, return_st=False):
    if not type(structure) is Bio.PDB.Structure.Structure:
        if '.pdb' in structure:
            parser = PDBParser(QUIET=True)
        else:
            parser = MMCIFParser(QUIET=True, auth_chains=auth_chains)

        if '.gz' in structure:
            try:
                with gzip.open(structure, 'rt') as handle:
                    structure = parser.get_structure('protein', handle)
            except:
                return None
        else:
            structure = parser.get_structure('protein', structure)

    chains = []
    if model_id is not None:
        model = structure[model_id]
        for chain in model:
            residues = [res for res in chain.get_residues()
                        if is_aa(res, standard=True)]
            if len(residues) >= min_residues:  # 根据含有的最小残基数量来确定是否排除链ID
                chains.append(chain.id)
    else:
        for model in structure:
            for chain in model:
                residues = [res for res in chain.get_residues()
                            if is_aa(res, standard=True)]
                if len(residues) >= min_residues:  # 根据含有的最小残基数量来确定是否排除链ID
                    chains.append(chain.id)
                # else:
                #     # 避免残基全为UNK时被排除
                #     residues = [res for res in chain if res.id[0] == ' ']
                #     all_unk = bool(residues) and all(res.resname.strip() == 'UNK' for res in residues)
                #     if all_unk:
                #         chains.append(chain.id)
    if return_st:
        return chains, structure
    else:
        return chains

def get_rcsb_pdb_file(file_path):
    if not os.path.exists(file_path):
        file_name = file_path.split('/')[-1]
        response = get_response(rcsb_pdb_download_url + file_name)
        with open(file_path, 'w') as f:
            f.write(response.text)


def get_rcsb_pdb_and_assm_file(file_path):
    """
    确认并下载指定的结构文件，并检查是否为对称生物组装体
    """
    file_name = file_path.split('/')[-1]
    if not os.path.exists(file_path):
        response = get_response(rcsb_pdb_download_url + file_name)
        with open(file_path, 'w') as f:
            f.write(response.text)

    header_dict = MMCIF2Dict(file_path)
    assembly_list = [file_name]
    if len(header_dict['_pdbx_struct_assembly_gen.assembly_id']) > 1 or \
            len(header_dict['_pdbx_struct_assembly_gen.oper_expression'][0].split(',')) > 1:  # 该组装体有多种对称情况
        assembly_list = []  # 清空原有的文件名称，只添加组装体
        assembly_gen_assembly_ids = list(set(header_dict['_pdbx_struct_assembly_gen.assembly_id']))
        assembly_gen_oper_expressions = header_dict['_pdbx_struct_assembly_gen.oper_expression']
        for assembly_gen_assembly_id in assembly_gen_assembly_ids:
            assembly_file_name = '%s-assembly%s.cif.gz' % (file_name[:4], assembly_gen_assembly_id)
            assembly_list.append(assembly_file_name)
            assembly_rcsb_path = file_path.replace(file_name, assembly_file_name)
            if not os.path.exists(assembly_rcsb_path):
                response = get_response(rcsb_pdb_download_url + assembly_file_name)
                with open(assembly_rcsb_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):  # 使用流式下载
                        # 过滤掉保持连接的空块
                        if chunk:
                            f.write(chunk)

    return assembly_list

def get_pdb_model_id(pdb_file):
    # 获取结构文件中的model的id（一般在组装体和NMR中使用）
    if '.pdb' in pdb_file or 'pdb1' in pdb_file:
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)

    if '.gz' in pdb_file:
        with gzip.open(pdb_file, 'rt') as handle:
            structure = parser.get_structure('protein', handle)
    else:
        structure = parser.get_structure('protein', pdb_file)

    model_ids = []
    for model in structure:
        model_ids.append(model.id)

    return model_ids

def count_chain_residues(structure, chain_id, model_id=0, standard=True):
    """
    统计指定的链中含有的残基个数
    """
    try:
        chain = structure[model_id][chain_id]
    except KeyError:
        raise ValueError(f"找不到模型 {model_id} 或链 {chain_id}")

    residues = [res for res in chain.get_residues()
                if not standard or is_aa(res, standard=True)]
    return len(residues)

def mol2_subst_names(path):
    subst_names = []
    with open(path) as fh:
        in_atom_block = False
        for line in fh:
            tag = line.strip()
            if tag == "@<TRIPOS>ATOM":
                in_atom_block = True
                continue
            if in_atom_block:
                if tag.startswith("@<TRIPOS>"):
                    break
                fields = tag.split()
                if len(fields) >= 8:
                    subst_names.append(fields[7])
    return subst_names

def chain_residue_index_range(structure, chain_id, model_id=0):
    """
    寻找pdb文件中指定的链的残基起始索引和结束索引
    """
    try:
        chain = structure[model_id][chain_id]
    except KeyError:
        raise ValueError(f"找不到模型 {model_id} 或链 {chain_id}")

    indices, detailed = [], []
    for residue in chain.get_residues():
        het_flag, seq_id, ins_code = residue.get_id()
        if het_flag != " ":   # 如需排除 HETATM，可加这行
            continue
        indices.append(seq_id)
        detailed.append((seq_id, ins_code))

    if not indices:
        raise ValueError(f"链 {chain_id} 没有符合条件的残基")

    seq_start, seq_end = min(indices), max(indices)
    detail_start, detail_end = min(detailed), max(detailed)
    return {
        "seq_range": (seq_start, seq_end),
        "detailed_range": (detail_start, detail_end)  # 含插入代码
    }

def rmsd_atoms(coords1, coords2):
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def get_ligand_coords_token(mol, is_sdf_str=False, removeHs=True):
    if type(mol) is not Chem.Mol:
        try:
            if is_sdf_str:
                mol = Chem.MolFromMolBlock(mol, removeHs=removeHs)
            else:
                mol = Chem.SDMolSupplier(mol, removeHs=removeHs)[0]
            conf = mol.GetConformer()  # 默认取第0个conformer
        except:
            return None, None
    else:
        try:
            conf = mol.GetConformer()
        except:
            return None, None
    atom_coords = []  # [atom_nums, 3]
    atomtypes = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        p = conf.GetAtomPosition(idx)
        if atom.GetSymbol() != 'H':
            atomtypes.append(atom.GetSymbol())
            atom_coords.append([float(p.x), float(p.y), float(p.z)])
    atom_coords = torch.tensor(atom_coords)
    atom_tokens = torch.tensor([aa2num[x] for x in atomtypes])

    return atom_coords, atom_tokens

def calc_binding_site_rmsd(apo_structure, holo_structure, binding_residues, apo_chain='A', holo_chain='A'):
    s1 = apo_structure[0][apo_chain]
    s2 = holo_structure[0][holo_chain]

    per_res_rmsd = {}
    for resi in binding_residues:
        if not (s1.has_id((' ', resi, ' ')) and s2.has_id((' ', resi, ' '))):
            continue
        r1 = s1[(' ', resi, ' ')]
        r2 = s2[(' ', resi, ' ')]
        atoms1 = np.array([a.coord for a in r1 if a.element != 'H'])
        atoms2 = np.array([a.coord for a in r2 if a.element != 'H'])
        if len(atoms1) != len(atoms2):
            continue
        per_res_rmsd[resi] = rmsd_atoms(atoms1, atoms2)

    if len(per_res_rmsd) > 0:
        pocket_rmsd = np.mean(list(per_res_rmsd.values()))
    else:
        pocket_rmsd = np.nan

    return per_res_rmsd, pocket_rmsd

def move_to_origin(pdb):
    cmd.reinitialize()
    cmd.load(pdb, "_all")
    # 对于组装体，合并所有可见原子到一个新对象，方便统一计算质心
    if '.zip' in pdb:
        cmd.create("_all", "_all")
    com = cmd.centerofmass("_all")
    neg_com = [-x for x in com]  # 构造反向平移向量 [ -x, -y, -z ]
    cmd.translate(neg_com, "_all")  # 移动蛋白质至几何中心

    # 获取字符串
    pdb_data = cmd.get_pdbstr("_all")
    return pdb_data


def kabsch_align(P, Q):
    """
    使用 PyTorch 实现 Kabsch 算法。
    """

    assert P.shape == Q.shape, "张量形状必须一致"

    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    H = torch.matmul(P_centered.T, Q_centered)

    # 奇异值分解 SVD
    U, S, Vh = torch.linalg.svd(H)

    # 计算旋转矩阵 R
    R = torch.matmul(Vh.T, U.T)

    # 检查是否发生反射 (手性反转)
    if torch.linalg.det(R) < 0:
        Vh_fixed = Vh.clone()
        Vh_fixed[2, :] = Vh_fixed[2, :] * -1
        R = torch.matmul(Vh_fixed.T, U.T)

    # 计算平移向量 t
    t = centroid_P - torch.matmul(centroid_Q, R.T)

    # 应用变换得到对齐后的坐标
    aligned_Q = torch.matmul(Q, R.T) + t

    return aligned_Q, R, t

# 计算对齐对应链的全原子RMSD
def calc_all_atom_rmsd(holo_pdb, apo_pdb, chain_h='A', chain_a='A'):
    cmd.reinitialize()
    cmd.load(holo_pdb, "holo")
    cmd.load(apo_pdb, "hit")
    rmsd = cmd.align(f"hit and chain {chain_a}",
                       f"holo and chain {chain_h}")[0]
    # cmd.quit()
    return rmsd

# 将蛋白质的其中一条链对齐到另外一个蛋白质的其中一条链，并保存对齐后的结构
def align_chain(apo_pdb, holo_pdb, chain_a, chain_h, save_file='apo_aligned.pdb'):
    cmd.reinitialize()
    cmd.load(apo_pdb, "apo")
    cmd.load(holo_pdb, "holo")
    cmd.align(f"apo and chain {chain_a}",
              f"holo and chain {chain_h}")
    cmd.save(f"{save_file}", "apo")


def compute_binding_site_chi_diff(holo_pdb, apo_pdb, binding_residues, apo_chain='A', holo_chain='A'):
    """
    计算指定结合位点残基的 χ1, χ2, χ3, χ4 角差（单位°）
    """
    # 加载结构
    holo = md.load_pdb(holo_pdb)
    apo = md.load_pdb(apo_pdb)

    holo_chain_ids = [chain.chain_id for chain in holo.topology.chains]
    apo_chain_ids = [chain.chain_id for chain in apo.topology.chains]

    results = {}

    # 最多计算到chi4
    for chi_fn in [md.compute_chi1, md.compute_chi2, md.compute_chi3, md.compute_chi4]:
        # 计算对应的二面角并获取使用了哪些原子来计算(*_chi_indices)
        holo_chi_indices, holo_chi_list = chi_fn(holo)
        apo_chi_indices, apo_chi_list = chi_fn(apo)

        for site in binding_residues:
            apo_select_str = "resSeq %s and chainid %s" % (site, apo_chain_ids.index(apo_chain))
            holo_select_str = "resSeq %s and chainid %s" % (site, holo_chain_ids.index(holo_chain))
            """mdtraj的链ID实际上是以数字索引的形式存储"""
            holo_atom_idx = holo.topology.select(holo_select_str)  # 获取结合位点的所有原子编号
            apo_atom_idx = apo.topology.select(apo_select_str)

            # 获取结合位点的chi角
            has_chi = False
            for chi_idx, holo_idx in enumerate(holo_chi_indices):
                res_idx = holo.topology.atom(holo_idx[1]).residue.resSeq  # 第二个原子必定是当前残基。对于含有插入码的氨基酸只会返回数字部分
                if res_idx == site and np.isin(holo_idx, holo_atom_idx).all():  # 当前蛋白质可能为同源寡聚体，因此还需要保证计算二面角的原子的编号属于对应的残基
                    has_chi = True
                    break
            if has_chi:  # 防止因为某些氨基酸没有二面角从而计算错误
                holo_chi = holo_chi_list[0][chi_idx]
                for chi_idx, apo_idx in enumerate(apo_chi_indices):
                    res_idx = apo.topology.atom(apo_idx[1]).residue.resSeq
                    if res_idx == site:
                        break
                apo_chi = apo_chi_list[0][chi_idx]

                diff = np.rad2deg(np.abs((holo_chi - apo_chi + np.pi) % (2 * np.pi) - np.pi))
                results.setdefault(site, []).append(diff)
            else:
                results.setdefault(site, []).append(float('inf'))

    return results

def align_seq(target_seq, query_seq, mode='global', return_only_alignment=False):
    # 序列相似性比对器
    aligner = PairwiseAligner()
    aligner.mode = mode  # 设置对比模式
    aligner.match_score = 2
    aligner.mismatch_score = -2
    # 让开 gap 很贵，避免为了解决首位不匹配而开一个 gap
    aligner.open_gap_score = -12
    aligner.extend_gap_score = -2
    try:
        alignment = aligner.align(target_seq, query_seq)[0]  # 取第一个最优比对
        if return_only_alignment:
            return alignment
        # 计算一致性百分比 (Identity Percentage)
        identity_count = alignment.counts().identities
        alignment_length = alignment.length
        identity_percent = (identity_count / alignment_length)  # 相似性得分（0-1）
        seq1_covered_len = sum(end - start for start, end in alignment.aligned[0])
        coverage_pct = (seq1_covered_len / len(target_seq))  # 覆盖度得分（0-1）
        return alignment, identity_count, identity_percent, coverage_pct
    except:
        return None, 0, 0., 0.

def read_seq_from_pdb(pdb_file):
    """读取pdb上的氨基酸序列（实际解析的序列）"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("my_structure", pdb_file)
    sequences = {}
    # 遍历所有残基
    for model in structure:
        for chain in model:
            seq = ''
            for residue in chain:
                # 过滤掉水分子和非标准残基（HETATM），只保留氨基酸
                if is_aa(residue, standard=True):
                    res_name = residue.get_resname()
                    seq += seq1(res_name)
            if seq != '':
                sequences[chain.id] = seq
    return sequences

def get_HETATM(pdb_path):
    """获取所有配体（除水分子）的信息"""
    if '.pdb' in pdb_path or 'pdb1' in pdb_path:
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)

    if '.gz' in pdb_path:
        with gzip.open(pdb_path, 'rt') as handle:
            structure = parser.get_structure('protein', handle)
    else:
        structure = parser.get_structure('protein', pdb_path)

    ligand_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # 获取 hetero_flag (元组的第一个元素)
                het_flag = residue.id[0]

                # 过滤条件：不是标准残基 (' ') 且 不是水 ('W')
                if het_flag != ' ' and het_flag != 'W':

                    # 有些时候水也会被标记为 H_HOH，为了保险可以双重检查残基名
                    if residue.get_resname() in ['HOH', 'WAT']:
                        continue
                    # 获取信息
                    ligand_list.append(residue.get_resname())
    return ligand_list

def get_ligand(pdb_path, ligand_name):
    """获取指定配体的坐标信息"""
    if '.pdb' in pdb_path or 'pdb1' in pdb_path:
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)

    if '.gz' in pdb_path:
        with gzip.open(pdb_path, 'rt') as handle:
            structure = parser.get_structure('protein', handle)
    else:
        structure = parser.get_structure('protein', pdb_path)

    ligand_dict = {}

    for model in structure:
        for chain in model:
            ligands = []
            for residue in chain:
                res_id = residue.id
                hetero_flag = res_id[0]

                if hetero_flag.startswith('H_%s' % ligand_name):
                    ligands.append(residue)
            ligand_dict['%s_%s' % (model.id, chain.id)] = ligands
    return ligand_dict

def get_protein_res(pdb_file, res_nb, chain_id, model_id=0):
    """
    返回指定结构中的某个链的某个残基
    """
    if 'cif' in pdb_file:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    if '.gz' in pdb_file:
        with gzip.open(pdb_file, 'rt') as handle:
            structure = parser.get_structure('protein', handle)
    else:
        structure = parser.get_structure('protein', pdb_file)

    chain = structure[model_id][chain_id]

    return chain[(' ', res_nb, ' ')]


def get_amino_acid_coords(structure, chain_id, model_id=0, max_atom_num=14, is_string=False, ignore_OXT=True):
    if type(structure) is str:
        if is_string:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("X", StringIO(structure))
        else:
            if 'cif' in structure:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            if '.gz' in structure:
                with gzip.open(structure, 'rt') as handle:
                    structure = parser.get_structure('protein', handle)
            else:
                structure = parser.get_structure('protein', structure)

    model = structure[model_id]

    if chain_id not in model:
        return []

    chain = model[chain_id]

    coords_list = []
    bfac_list = []  # 温度因子。反映了原子在晶体结构中的热运动或灵活性，数值越高，原子位置越不稳定，灵活性越高。
    occ_list = []  # 占有率。表示该原子出现在该位置的概率。
    res_list = []

    atom_mask = []

    for residue in chain:
        coords = np.zeros((max_atom_num, 3))
        bfac = np.full(max_atom_num, np.nan)
        occ = np.zeros(max_atom_num)
        mask = np.zeros(max_atom_num)
        # 只处理标准氨基酸 (忽略水分子 HOH 和其他配体)。通过 residue.id[0] == ' ' 来判断是否为标准残基
        if residue.id[0] == ' ':
            # 遍历该残基下的所有原子
            for atom in residue:
                if atom.id == 'OXT' and ignore_OXT:  # 常代表残基的羰基氧原子，特别是在C末端氨基酸的羧基，建模深度学习模型时通常会会忽略
                    continue
                try:
                    atom_index = aa2long_noblank[one_letter.index(seq1(residue.resname))].index(atom.fullname)  # cif文件读取原子时不会有空格
                except:
                    try:
                        atom_index = aa2long[one_letter.index(seq1(residue.resname))].index(atom.fullname)
                    except:
                        continue
                if atom_index >= max_atom_num:  # 最多取设置的最大原子数的坐标
                    break
                coords[atom_index] = atom.get_coord()
                bfac[atom_index] = atom.bfactor
                occ[atom_index] = atom.occupancy
                mask[atom_index] = 1.
            coords_list.append(coords)
            bfac_list.append(bfac)
            occ_list.append(occ)
            atom_mask.append(mask)
            res_list.append(residue)

    coords_list = np.array(coords_list)
    bfac_list = np.array(bfac_list)
    occ_list = np.array(occ_list)
    atom_mask = np.array(atom_mask)
    return res_list, coords_list, atom_mask, bfac_list, occ_list

def get_pdb_path(root_path, pdb_id):
    # 获取结构文件，如果本地不存在则前往rcsb上下载
    if not os.path.exists(root_path + '/%s.pdb' % pdb_id):
        if not os.path.exists(root_path + '/%s.cif' % pdb_id):
            response = get_response(rcsb_pdb_download_url + pdb_id + '.pdb')
            if response.status_code != 200:
                response = get_response(rcsb_pdb_download_url + pdb_id + '.cif')
                structure_path = root_path + '/%s.cif' % pdb_id
                with open(structure_path, 'w') as f:
                    f.write(response.text)
            else:
                structure_path = root_path + '/%s.pdb' % pdb_id
                with open(structure_path, 'w') as f:
                    f.write(response.text)
        else:
            structure_path = '../candidate_apo_pdb/%s.cif' % pdb_id
    else:
        structure_path = '../candidate_apo_pdb/%s.pdb' % pdb_id

    return structure_path

# 获取PDB上的共价键信息
def get_pdb_links(file_path):
    link_records = []
    if '.pdb' in file_path:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("LINK"):
                    # 根据 PDB 官方格式定义列宽
                    record = {
                        "at_name1": line[12:16].strip(),
                        "alt_loc1": line[16].strip(),
                        "res_name1": line[17:20].strip(),
                        "chain_id1": line[21].strip(),
                        "res_seq1": int(line[22:26].strip()),
                        "ins_code1": line[26].strip(),

                        "at_name2": line[42:46].strip(),
                        "alt_loc2": line[46].strip(),
                        "res_name2": line[47:50].strip(),
                        "chain_id2": line[51].strip(),
                        "res_seq2": int(line[52:56].strip()),
                        "ins_code2": line[56].strip(),

                        "length": line[73:78].strip()
                    }
                    link_records.append(record)
    else:
        # 将CIF文件读入字典
        mmcif_dict = MMCIF2Dict(file_path)
        if '_struct_conn.id' in mmcif_dict:
            for i in range(len(mmcif_dict['_struct_conn.id'])):
                if mmcif_dict['_struct_conn.conn_type_id'][i] == 'covale':
                    record = {
                        "at_name1": mmcif_dict['_struct_conn.ptnr1_label_atom_id'][i],
                        "alt_loc1": mmcif_dict['_struct_conn.pdbx_ptnr1_label_alt_id'][i],
                        "res_name1": mmcif_dict['_struct_conn.ptnr1_auth_comp_id'][i],
                        "chain_id1": mmcif_dict['_struct_conn.ptnr1_auth_asym_id'][i],
                        "res_seq1": mmcif_dict['_struct_conn.ptnr1_auth_seq_id'][i],
                        "ins_code1": mmcif_dict['_struct_conn.pdbx_ptnr1_PDB_ins_code'][i],

                        "at_name2": mmcif_dict['_struct_conn.ptnr2_label_atom_id'][i],
                        "alt_loc2": mmcif_dict['_struct_conn.pdbx_ptnr2_label_alt_id'][i],
                        "res_name2": mmcif_dict['_struct_conn.ptnr2_auth_comp_id'][i],
                        "chain_id2": mmcif_dict['_struct_conn.ptnr2_auth_asym_id'][i],
                        "res_seq2": mmcif_dict['_struct_conn.ptnr2_auth_seq_id'][i],
                        "ins_code2": mmcif_dict['_struct_conn.pdbx_ptnr2_PDB_ins_code'][i],

                        "length": mmcif_dict['_struct_conn.pdbx_dist_value'][i]
                    }
                    link_records.append(record)
    return link_records

def find_nearest_atom(protein_coords, target_coord):
    """
    寻找张量中距离给定坐标最近的原子的索引
    """
    # 利用广播机制直接相减，并计算最后一个维度（x,y,z）的 L2 范数（欧氏距离）
    distances = torch.norm(protein_coords - target_coord, dim=-1)

    # 找到整个 (L, 27) 矩阵中最小值的“展平”索引 (flattened index)
    min_flat_idx = torch.argmin(distances)

    # 将展平的索引还原回二维坐标 (残基索引, 原子索引)
    res_idx, atom_idx = torch.unravel_index(min_flat_idx, distances.shape)

    min_distance = distances[res_idx, atom_idx]

    return res_idx.item(), atom_idx.item(), min_distance.item()


def match_and_map_ligands(mol_pdbbind, mol_pdb):
    """
    将 PDB 中残缺的配体(mol_pdb) 匹配到 PDBBind 的完整配体(mol_pdbbind) 上。

    返回:
        match_dict: {pdbbind_atom_index: pdb_atom_index} 的映射字典
    """
    # 1. 寻找最大公共子结构 (MCS)
    # 关键参数：bondCompare=CompareAny (无视 PDB 糟糕的单双键分配)
    # 关键参数：completeRingsOnly=False (允许匹配残缺的环)
    mcs_res = rdFMCS.FindMCS(
        [mol_pdbbind, mol_pdb],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        timeout=10  # 防止极端庞大分子的死循环
    )

    if mcs_res.canceled:
        return None

    # 2. 将 MCS 的 SMARTS 转换为 RDKit Mol 对象
    mcs_mol = Chem.MolFromSmarts(mcs_res.smartsString)

    # 3. 分别在两个分子中获取 MCS 的匹配索引
    # GetSubstructMatch 返回的是一个元组，顺序与 mcs_mol 的原子顺序严格对应
    match_pdbbind = mol_pdbbind.GetSubstructMatch(mcs_mol)
    match_pdb = mol_pdb.GetSubstructMatch(mcs_mol)

    if not match_pdbbind or not match_pdb:
        return None

    # 4. 构建映射字典 { PDBBind 中的索引 : PDB 中的索引 }
    index_mapping = {}
    for i in range(len(match_pdbbind)):
        idx_bind = match_pdbbind[i]
        idx_pdb = match_pdb[i]
        index_mapping[idx_bind] = idx_pdb

    return index_mapping


def matching_pdbbind_rcsb_protein(pdbbind_path, rcsb_path, pdbbind_auth_chains=True, rcsb_auth_chains=True):
    pdbbind_chains, pdbbind_st = get_protein_chains(pdbbind_path, return_st=True, auth_chains=pdbbind_auth_chains)
    rcsb_chains, rcsb_st = get_protein_chains(rcsb_path, return_st=True, auth_chains=rcsb_auth_chains)

    pdb_id = pdbbind_path.split('/')[-1][:4]

    header_dict = MMCIF2Dict(rcsb_path)
    rcsb_biounit_id = 1  # 用于记录生物组装体的id（无论是否为生物组装体默认都为1）
    assembly_rcsb_path = rcsb_path
    # 确认是否为生物组装体
    if len(header_dict['_pdbx_struct_assembly_gen.assembly_id']) > 1 and len(pdbbind_chains) > len(rcsb_chains):  # 该组装体有多种对称情况
        assembly_gen_assembly_ids = list(set(header_dict['_pdbx_struct_assembly_gen.assembly_id']))
        for assembly_gen_assembly_id in assembly_gen_assembly_ids:
            assembly_rcsb_path = rcsb_path.replace('%s.cif' % pdb_id, '%s-assembly%s.cif.gz' % (pdb_id, assembly_gen_assembly_id))
            with gzip.open(assembly_rcsb_path, 'rb') as f_in:
                with open('./temp.cif', 'wb') as f_out:
                    f_out.write(f_in.read())
            result = subprocess.run(['../MMalign', pdbbind_path, './temp.cif', '-infmt2', '3'], capture_output=True, text=True)  # 执行对比
            name = mm_name1_patthen.findall(result.stdout)[0]
            rmsd = float(number_pattern.findall(rmsd_pattern.findall(result.stdout)[0])[0])
            if '::' not in name[0] and '::' not in name[1] and rmsd == 0.:  # 避免因对称性导致后续的问题
                rcsb_chains, rcsb_st = get_protein_chains(assembly_rcsb_path, return_st=True, auth_chains=rcsb_auth_chains)
                rcsb_biounit_id = assembly_gen_assembly_id
                break
    elif len(header_dict['_pdbx_struct_assembly_gen.oper_expression'][0].split(',')) > 1 and len(pdbbind_chains) > len(rcsb_chains):  # 该组装体只有一种对称情况
        rcsb_chains, rcsb_st = get_protein_chains(rcsb_path.replace('%s.cif' % pdb_id, '%s-assembly1.cif.gz' % pdb_id), return_st=True, auth_chains=rcsb_auth_chains)

    return pdbbind_chains, pdbbind_st, rcsb_chains, rcsb_st, rcsb_biounit_id, assembly_rcsb_path


def get_cif_label_auth_mapping(cif_path):
    """
    获取cif文件中label和auth的id的映射
    """
    cif_dict = MMCIF2Dict(cif_path)
    # 获取官方链ID列表和对应的作者链ID列表
    label_ids = cif_dict.get("_pdbx_poly_seq_scheme.asym_id", [])
    auth_ids = cif_dict.get("_pdbx_poly_seq_scheme.pdb_strand_id", [])

    if not label_ids or not auth_ids:
        chain_mapping = {}
    else:
        chain_mapping = dict(zip(auth_ids, label_ids))  # {auth_id: label_id}

    return chain_mapping


if __name__ == '__main__':
    # seq = read_seq_from_pdb('/srv/storage1/ssd/zzl/lzj/PDBbind+/PDBbind_v2024_PL/2001-2010/2g96/2g96_pocket.pdb')

    # identity_count, identity_percent, coverage_pct = align_seq(
    #     'MHHHHHHSSGRENLYFQGTSKLKYVLQDARFFLIKSNNHENVSLAKAKGVWATLPVNEKKLNLAFRSARSVILIFSVRESGKFQGFARLSSESHHGGSPIHWVLPAGMSAKMLGGVFKIDWICRRELPFTKSAHLTNPWNEHKPVKIGRDGQEIELECGTQLCLLFPPDESIDLYQVIHKMRH',
    #     'MHHHHHHSSGRENLYFQGTSKLKYVLQDARFFLIKSNNHENVSLAKAKGVWSTLPVNEKKLNLAFRSARSVILIFSVRESGKFQGFARLSSESHHGGSPIHWVLPAGMSAKALGGVFKIDWICRRELPFTKSAHLTNPWNEHKPVKIGRDGQEIELECGTQLCLLFPPDESIDLYQVIHKMRH',
    # )

    # ligand_list = get_HETATM('/home/a1356913256/python/molecular_generation/APO2MolFlow/data/candidate_apo_pdb/2H75.pdb')

    # residue_idx, coords_list, atom_mask = get_amino_acid_coords('/home/a1356913256/python/molecular_generation/APO2MolFlow/data/candidate_apo_pdb/2H75.pdb', chain_id='A')

    # a = get_pdb_links('/data/user1/data/APO2MolFlow/7S6W.cif')

    a = get_rcsb_pdb_and_assm_file('/root/autodl-tmp/dataset/APO2MolFlow/apo/candidate/pdb/9MR8.cif')
    print()
