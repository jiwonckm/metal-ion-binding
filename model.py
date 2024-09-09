import torch
import torchvision
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MultilabelAveragePrecision
import wandb
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
from margin import MarginScheduledLossFunction

ions = ['Ca', 'Co', 'Cu', 'Fe2', 'Fe', 'Mg', 'Mn', 'PO4', 'SO4', 'Zn', 'Null']

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

    def forward(self, h_V):
        mask = None
        # Self-attention
        dh = self.attention(h_V, h_V, h_V, mask)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        return h_V


class IonBindingModel(nn.Module):
    def __init__(self, feature_dim=2560, hidden_dim=128, num_encoder_layers=4, num_heads=4, augment_eps=0.05, dropout=0.2):
        super(IonBindingModel, self).__init__()

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
        self.FC_NULL_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_NULL_2 = nn.Linear(hidden_dim, 1, bias=True)
  
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, protein_feat):
        return self.classify(protein_feat)

    def project(self, protein_feat):
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(protein_feat)

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V)

        return h_V

    def classify(self, protein_feat):
        h_V = self.project(protein_feat)
        
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
        logits_NULL = self.FC_NULL_2(F.leaky_relu(self.FC_NULL_1(h_V)))

        logits = torch.cat((logits_CA, logits_CO, logits_CU, logits_FE2, logits_FE, logits_MG, logits_MN, logits_PO4, logits_SO4, logits_ZN, logits_NULL), 1)
        
        return logits
        