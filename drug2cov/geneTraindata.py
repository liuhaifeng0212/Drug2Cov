import pandas as pd



drug_path = 'dataset/drug_dict'
disease_path = 'dataset/disease_dict'
protein_path = 'dataset/protein_dict'
se_path = 'dataset/se_dict'
drug_diseas = 'dataset/se_dict'

def getId(path):
    idmap = {}
    with open(path,'r') as f:
        for line in f:
            id_,name = line.strip().split(':')
            idmap[id_]=name
    return idmap
drug_map = getId(drug_path)
disease_map = getId(disease_path)
protein_map = getId(protein_path)
se_map = getId(se_path)


df = pd.read_csv()