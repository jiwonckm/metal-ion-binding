import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import esm
import ast
import pickle as pk
from tqdm import tqdm
import lightning as L

class MionicDataset(Dataset):
    def __init__(self, csv_file, rep_dir, truth_dir):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.sample(n=500, replace=False)   # COMMENT OUT BEFORE ACTUAL RUN
        self.truth_vals = pk.load(open(truth_dir, 'rb'))
        self.rep_dir = rep_dir

        self.ids = []  # length: (protein * residue)
        self.prot_rep = {}
        self.labels = {}
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            ID = row['ID']
            Seq = row['Seq']
            
            # create dictionary for protein level
            self.prot_rep[ID] = torch.load(self.rep_dir + f'{ID}.pt')['representations'][33]
            #self.prot_rep[ID] = torch.tensor([[0]*1280 for _ in range(len(Seq))], dtype=torch.float32)
            
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
                    labels.append(0)
                else:
                    labels = [0]*10
                    labels.append(1)
                labels = torch.tensor(labels, dtype=torch.float32)
         
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
        
def collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features, 0)
    labels = torch.stack(labels, 0)
    
    return features, labels
    
class MionicDatamodule(L.LightningDataModule):
    def __init__(self, data_dir, rep_dir, truth_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.rep_dir = rep_dir
        self.truth_dir = truth_dir
        self.batch_size = batch_size
        
    def setup(self):
        dataset = MionicDataset(self.data_dir, self.rep_dir, self.truth_dir)
        self.train_set, self.val_set, self.test_set = random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))

        # Balance the data
        train_labels = torch.stack([self.train_set.dataset.labels[self.train_set.dataset.ids[i]] for i in self.train_set.indices])
        negative_count = torch.sum(torch.all(train_labels == 0, dim=1))   # number of negative labels
        negative_weight = 0.5 / negative_count
        ion_count = torch.sum(train_labels, dim=0)                        # 10-d vector of count of each ions
        ion_weight = 0.05 / ion_count
        
        weights = []
        for row in train_labels:
            weight = torch.sum(row * ion_weight)
            if weight==0: 
                weight = negative_weight
            weights.append(weight)
        weights = torch.tensor(weights)
        self.sampler = WeightedRandomSampler(weights=weights, num_samples=len(self.train_set), replacement=False)

        return ion_weight

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=collate_fn, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=collate_fn)

if __name__ == '__main__':
    for batch in MionicDataset('../../Project/mionic_dataset/mionic_train.csv', '../../Project/mionic_dataset/train_embeddings/', '../../Project/m-ionic/data/pos_data/multi_ion.pkl'):
        inputs, labels = batch
        
        break