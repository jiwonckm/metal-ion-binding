import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
import esm
import ast
import pickle as pk
from tqdm import tqdm
import lightning as L
import itertools
import numpy as np
import random

ions = ['CA','CO','CU','FE2', 'FE', 'MG', 'MN', 'PO4', 'SO4', 'ZN']

class MionicDataset(Dataset):
    def __init__(self, csv_file, rep_dir, truth_dir, num_samples=None):
        self.data = pd.read_csv(csv_file)
        if num_samples:
            self.data = self.data.sample(n = num_samples, replace=False)   # COMMENT OUT BEFORE ACTUAL RUN. total sample num = 21736
        self.truth_vals = pk.load(open(truth_dir, 'rb'))
        self.rep_dir = rep_dir

        self.ids = []  # length: (protein * residue). all combinations of sequence id + residue number
        self.prot_rep = {} # key: sequence id, val: ESM2 representation
        self.labels = {} # key: sequence id + residue number, val: binary vector
        self.reverse_labels = {'NULL':[]} # key: each ion, val: sequence id + residue that binds to that ion
        
        for ion in ions:
            self.reverse_labels[ion] = []
        for index, row in tqdm(self.data.iterrows(), total=len(self.data)):
            ID = row['ID']
            Seq = row['Seq']
            
            # create dictionary for protein level
            self.prot_rep[ID] = torch.load(self.rep_dir + f'{ID}.pt')['representations'][33]
            
            for i in range(len(Seq)):
                self.ids.append(f'{ID}_{i}')

                if ID in self.truth_vals: # True if this protein binds to an ion at all
                    binding_ions = self.truth_vals[ID]['ions']
                    labels = []
                    for ion in ions:
                        if ion not in binding_ions:
                            labels.append(0)
                        else:
                            labels.append(int(binding_ions[ion][i]))
                            if int(binding_ions[ion][i]): self.reverse_labels[ion].append(f'{ID}_{i}')
                                
                if sum(labels)==0:
                    labels.append(1)
                    self.reverse_labels['NULL'].append(f'{ID}_{i}')
                else: labels.append(0)
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
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def setup(self):
        
        # Balance the data
        train_labels = torch.stack(list(self.dataset.labels.values()), dim=0)
        ion_count = torch.sum(train_labels, dim=0)                        # 11-d vector of count of each ions
        ion_weight = 1 / ion_count
        
        weights = []
        for row in train_labels:
            weight = torch.sum(row * ion_weight)
            weights.append(weight)
        weights = torch.tensor(weights)
        self.sampler = WeightedRandomSampler(weights=weights, num_samples=len(self.dataset), replacement=True)

        # return ion_weight

    def dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn, sampler=self.sampler)

class ConDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.reverse_labels = dataset.reverse_labels
        self.null_residues = random.sample(dataset.reverse_labels['NULL'], 100)
        
        # Create pairs of (residue1, residue2) for each ion type excluding NULL
        self.pairs = []
        for ion, residues in dataset.reverse_labels.items():
            if ion != "NULL":
                self.pairs.extend(itertools.combinations(residues, 2))
        # print(len(self.pairs)) # when I ran with 3000 samples, the length of this was 28610987.
        self.pairs = random.sample(self.pairs, 5000)
        
        # Create all possible tuples by combining pairs with NULL residues
        self.batch = list(itertools.product(self.pairs, self.null_residues))
        
    def __len__(self):
        return len(self.batch)

    def __getitem__(self, idx):
        pair, null_residue = self.batch[idx]
        batch = []
        for id in (pair[0], pair[1], null_residue):
            last_underscore_index = id.rfind('_')
            residue = int(id[last_underscore_index+1:])
            id = id[:last_underscore_index]
        
            protein_rep = self.dataset.prot_rep[id]         # shape: (residue #) x (1280)
            residue_rep = protein_rep[residue, :]   # (1280)
            feature = torch.cat((residue_rep, torch.mean(protein_rep, dim=0)), dim=0)   # (1280 + 1280)
            batch.append(feature)
            
        return batch

class ConDatamodule(L.LightningDataModule):
    def __init__(self, dataset, batch_size, shuffle, con_num_samples):
        super().__init__()
        self.dataset = ConDataset(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.con_num_samples = con_num_samples
        
    def setup(self):
        # there are too many samples, so we are randomly selecting con_num_samples
        indices = np.random.choice(len(self.dataset), self.con_num_samples, replace=False) 
        self.sampled_dataset = Subset(self.dataset, indices)

    def dataloader(self):
        return DataLoader(self.sampled_dataset, batch_size=self.batch_size)


if __name__ == '__main__':
    for batch in MionicDataset('../../Project/mionic_dataset/mionic_train.csv', '../../Project/mionic_dataset/train_embeddings/', '../../Project/m-ionic/data/pos_data/multi_ion.pkl'):
        inputs, labels = batch
        
        break