import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import esm
import ast
import pickle as pk

class MionicDataset(Dataset):
    def __init__(self, csv_file, rep_dir, truth_dir):
        self.data = pd.read_csv(csv_file)
        self.truth = pk.load(open(truth_dir, 'rb'))
        self.rep_dir = rep_dir

        self.ids = []
        for index, row in self.data.iterrows():
            ID = row['ID']
            Seq = row['Seq']
            for i in range(len(Seq)):
                self.ids.append(f'{ID}_{i}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        last_underscore_index = id.rfind('_')
        residue = int(id[last_underscore_index+1:])
        id = id[:last_underscore_index]
        
        protein_rep = torch.load(self.rep_dir + f'{id}.pt')['representations'][33] # shape: (residue #) x (1280)
        residue_rep = protein_rep[residue, :]   # (1280)
        feature = torch.cat((residue_rep, torch.mean(protein_rep, dim=0)), dim=0)   # (1280 + 1280)

        if id in self.truth:
            truths = self.truth[id]['ions']
            labels = []
            for ion in ['CA','CO','CU','FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN']:
                if ion not in truths:
                    labels.append(0)
                else:
                    labels.append(int(truths[ion][residue]))
        else:
            labels = [0] * 10
        labels = torch.tensor(labels, dtype=torch.float32)

        # labels = self.data.iloc[idx]['Label'] # Ca, Co, Cu, Fe2, Fe, Mg, Mn, PO4, SO4, Zn
        # labels = ast.literal_eval(labels)
        # labels = torch.tensor(labels, dtype=torch.float32)
        
        return feature, labels

if __name__ == '__main__':
    for batch in MionicDataset('../../Project/mionic_dataset/mionic_train.csv', '../../Project/mionic_dataset/train_embeddings/', '../../Project/m-ionic/data/pos_data/multi_ion.pkl'):
        inputs, labels = batch
        
        break