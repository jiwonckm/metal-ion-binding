import pandas as pd
import torch
from torch.utils.data import Dataset
import esm
import ast
import pickle as pk
from tqdm import tqdm

class MionicDataset(Dataset):
    def __init__(self, csv_file, rep_dir, truth_dir):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.sample(n=300, replace=False)   # COMMENT OUT BEFORE ACTUAL RUN
        self.truth_vals = pk.load(open(truth_dir, 'rb'))
        self.rep_dir = rep_dir

        self.ids = []  # length: (protein * residue)
        self.prot_rep = {}
        self.labels = {}
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            ID = row['ID']
            Seq = row['Seq']
            
            # create dictionary for protein level
            #self.prot_rep[ID] = torch.load(self.rep_dir + f'{ID}.pt')['representations'][33]
            self.prot_rep[ID] = torch.tensor([[0]*1280 for _ in range(len(Seq))], dtype=torch.float32)
            
            for i in range(len(Seq)):
                self.ids.append(f'{ID}_{i}')

                if ID in self.truth_vals:
                    truths = self.truth_vals[ID]['ions']
                    labels = []
                    for ion in ['CA','CO','CU','FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN']:
                        if ion not in truths:
                            labels.append(0)
                        else:
                            labels.append(int(truths[ion][i]))
                    labels = torch.tensor(labels, dtype=torch.float32)
                else:
                    labels = torch.zeros(10)
         
                self.labels[f'{ID}_{i}'] = labels


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        labels = self.labels[id]
        
        last_underscore_index = id.rfind('_')
        residue = int(id[last_underscore_index+1:])
        id = id[:last_underscore_index]
        
        protein_rep = self.prot_rep[id]         # shape: (residue #) x (1280)
        residue_rep = protein_rep[residue, :]   # (1280)
        feature = torch.cat((residue_rep, torch.mean(protein_rep, dim=0)), dim=0)   # (1280 + 1280)
        
        return feature, labels

if __name__ == '__main__':
    for batch in MionicDataset('../../Project/mionic_dataset/mionic_train.csv', '../../Project/mionic_dataset/train_embeddings/', '../../Project/m-ionic/data/pos_data/multi_ion.pkl'):
        inputs, labels = batch
        
        break