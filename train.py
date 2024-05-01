import torch
import torchvision
from datetime import datetime
from dataloader import MionicDataset
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MultilabelAveragePrecision
import wandb
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence

ions = ['Ca', 'Co', 'Cu', 'Fe2', 'Fe', 'Mg', 'Mn', 'PO4', 'SO4', 'Zn']

class Self_Attention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(num_hidden / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    def forward(self, q, k, v, mask=None):
        q = self.transpose_for_scores(q) # [bsz, heads, protein_len, hid]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            attention_mask = (1.0 - mask) * -10000
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)

        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        outputs = torch.matmul(attention_scores, v)

        outputs = outputs.permute(0, 2, 1).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        return outputs


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.leaky_relu(self.W_in(h_V))
        h = self.W_out(h)
        return h

class TransformerLayer(nn.Module):
    def __init__(self, num_hidden = 64, num_heads = 4, dropout = 0.2):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)])

        self.attention = Self_Attention(num_hidden, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, mask=None):
        # Self-attention
        dh = self.attention(h_V, h_V, h_V, mask)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask is not None:
            mask = mask.unsqueeze(-1)
            h_V = mask * h_V
        return h_V


class ProteinClassifier(nn.Module):
    def __init__(self, feature_dim=2560, hidden_dim=128, num_encoder_layers=4, num_heads=4, augment_eps=0.05, dropout=0.2):
        super(ProteinClassifier, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = nn.Sequential(
                                          nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # ion-specific layers
        # ['Ca', 'Co', 'Cu', 'Fe2', 'Fe', 'Mg', 'Mn', 'PO4', 'SO4', 'Zn']
        self.FC_CA_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CA_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CO_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CO_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CU_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CU_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_FE2_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_FE2_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_FE_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_FE_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MG_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MG_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MN_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MN_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_PO4_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_PO4_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_SO4_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_SO4_2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_ZN_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_ZN_2 = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, protein_feat, mask):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(protein_feat)

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)
        
        # ['Ca', 'Co', 'Cu', 'Fe2', 'Fe', 'Mg', 'Mn', 'PO4', 'SO4', 'Zn']
        logits_CA = self.FC_CA_2(F.leaky_relu(self.FC_CA_1(h_V)))
        logits_CO = self.FC_CO_2(F.leaky_relu(self.FC_CO_1(h_V)))
        logits_CU = self.FC_CU_2(F.leaky_relu(self.FC_CU_1(h_V)))
        logits_FE2 = self.FC_FE2_2(F.leaky_relu(self.FC_FE2_1(h_V)))
        logits_FE = self.FC_FE_2(F.leaky_relu(self.FC_FE_1(h_V))) 
        logits_MG = self.FC_MG_2(F.leaky_relu(self.FC_MG_1(h_V)))
        logits_MN = self.FC_MN_2(F.leaky_relu(self.FC_MN_1(h_V)))
        logits_PO4 = self.FC_PO4_2(F.leaky_relu(self.FC_PO4_1(h_V)))
        logits_SO4 = self.FC_SO4_2(F.leaky_relu(self.FC_SO4_1(h_V)))
        logits_ZN = self.FC_ZN_2(F.leaky_relu(self.FC_ZN_1(h_V)))

        logits = torch.cat((logits_CA, logits_CO, logits_CU, logits_FE2, logits_FE, logits_MG, logits_MN, logits_PO4, logits_SO4, logits_ZN), 1)
        
        return logits

# class ProteinClassifier(nn.Module):
#     def __init__(self, input_shape=2560, output_shape=10, embedding_dim=256):
#         super(ProteinClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_shape, 120)
#         self.fc2 = nn.Linear(120, 84)
        
#         self.cls_head = nn.Linear(84, output_shape)

#         self.ca_head = nn.Linear(84, 1)
#         self.co_head = nn.Linear(84, 1)
#         self.cu_head = nn.Linear(84, 1)
#         self.fe2_head = nn.Linear(84, 1)
#         self.fe_head = nn.Linear(84, 1)
#         self.mg_head = nn.Linear(84, 1)
#         self.mn_head = nn.Linear(84, 1)
#         self.po4_head = nn.Linear(84, 1)
#         self.so4_head = nn.Linear(84, 1)
#         self.zn_head = nn.Linear(84, 1)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # shape of x is (batch)x(2 * feature_dim)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.cls_head(x)
#         x = self.sigmoid(x)
#         return x

def collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features, 0)
    labels = torch.stack(labels, 0)
    
    return features, labels

def main(config):
    
    def train_one_epoch(epoch_index, device):
        metric = MultilabelAveragePrecision(num_labels=10, average=None, thresholds=None)
        metric.to(device)
        batch_metric = MultilabelAveragePrecision(num_labels=10, average=None, thresholds=None)
        batch_metric.to(device)
        
        running_loss = 0
        last_loss = 0
        
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs, mask=None)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            labels = labels.to(torch.int64)
            aupr = metric(outputs, labels)
            x = batch_metric(outputs, labels)
            
            if i%1000 == 999:
                last_loss = running_loss / 1000
                print('   batch {} loss: {}'.format(i+1, last_loss))
    
                batch_aupr = batch_metric.compute()
                x = {'train/batch_loss': last_loss}
                for i, ion in enumerate(ions):
                    x[f'train/batch_aupr_{ion}'] = float(batch_aupr[i].detach().cpu())
                
                wandb.log(x)
                
                running_loss = 0
                batch_metric.reset()
                
        aupr = metric.compute()
        return last_loss, aupr

    # Set CUDA device
    if (config.device == 'cpu') or (not torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{config.device}')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load DataModule
    dataset = MionicDataset(config.data_dir, config.rep_dir, config.truth_dir)
    train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    # Balance the data
    train_labels = torch.stack([train_set.dataset.labels[train_set.dataset.ids[i]] for i in train_set.indices])
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
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set), replacement=False)

    # Load DataLoaders
    batch_size = config.batch_size
    shuffle = config.shuffle
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    # Model and Optimizer
    model = ProteinClassifier()
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize wandb
    wandb.login()
    wandb.init(project=config.wandb_proj, name=config.run_name)

    # Metric
    val_metric = MultilabelAveragePrecision(num_labels=10, average=None, thresholds=None)
    val_metric.to(device)

    # Run training
    EPOCHS = config.epochs
    best_vloss = 1_000_000
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch))
    
        model.train(True)
        avg_loss, aupr = train_one_epoch(epoch, device)
        x = {'train/loss': avg_loss}
        for i, ion in enumerate(ions):
            x[f'train/aupr_{ion}'] = float(aupr[i])
    
        wandb.log(x)
    
        running_vloss = 0
        model.eval()
    
        val_metric.reset()
        
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs, mask=None)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                vlabels = vlabels.to(torch.int64)
                vaupr = val_metric(voutputs, vlabels)
    
                wandb.log({'val/loss': vloss})
    
        avg_vloss = running_vloss / (i+1)
        vaupr = val_metric.compute()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('AUPR train {} valid {}'.format(torch.mean(aupr).item(), torch.mean(vaupr).item()))

        x = {'val/avg_loss': avg_vloss, 'epoch':epoch}
        for i, ion in enumerate(ions):
            x[f'val/aupr_{ion}'] = float(vaupr[i])
        wandb.log(x)
    
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}/model_{}_{}'.format(config.model_save_dir, timestamp, epoch)
            torch.save(model.state_dict(), model_path)
    
    model.cpu()

if __name__ == "__main__":
    parser = ArgumentParser(description='Train')
    parser.add_argument('--run_name', type=str, help='Name of run')
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--device', required=True, help='Specify device')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.run_name = args.run_name
    config.device = args.device
    
    main(config)