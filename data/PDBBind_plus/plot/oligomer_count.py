import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import matplotlib
matplotlib.use('agg')

if __name__ == '__main__':
    oligomer_count = {}

    for file in os.listdir('../pdbbind_fasta/full_chain'):
        records = list(SeqIO.parse('../pdbbind_fasta/full_chain/%s' % file, 'fasta'))
        try:
            oligomer_count[len(records)] += 1
        except:
            oligomer_count[len(records)] = 1

    holo_num = sum(oligomer_count.values())
    oligomer_count = {oligomer: count / holo_num for oligomer, count in oligomer_count.items()}
    df = pd.DataFrame(list(oligomer_count.items()), columns=['Oligomer num', 'Frequency'])
    df = df.sort_index()

    plt.figure(figsize=(12, 6))  # 设置画布大小
    sns.set_theme(style="whitegrid")

    sns.barplot(data=df, x='Oligomer num', y='Frequency')

    plt.title("Oligomer num of PDBBind+")
    plt.xlabel("Oligomer num")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig('./pdbbind+_oligomer_num.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('./pdbbind+_oligomer_num.png', format='png', dpi=300, bbox_inches='tight')
