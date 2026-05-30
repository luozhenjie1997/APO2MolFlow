import pickle
import os
import torch
import gzip
import shutil
import copy
import pandas as pd
import pyarrow.parquet as pq
from PDBBind_plus.utils import get_protein_chains, load_config

if __name__ == '__main__':
    config, config_name = load_config('../config/base_config.yaml')

    a = pq.ParquetFile('/root/autodl-tmp/dataset/PLINDER/2024-06_v2_links_kind=apo_links.parquet').read().to_pandas()
    print()
