import torch
import torchvision
from datetime import datetime
from dataloader import MionicDataset
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MultilabelAveragePrecision
import wandb
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence

ions = ['Ca', 'Co', 'Cu', 'Fe2', 'Fe', 'Mg', 'Mn', 'PO4', 'SO4', 'Zn']

class ProteinClassifier(nn.Module):
    def __init__(self, input_shape=2560, output_shape=10, embedding_dim=256):
        super(ProteinClassifier, self).__init__()
        self.fc1 = nn.Linear(input_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.cls_head = nn.Linear(84, output_shape)

        self.ca_head = nn.Linear(84, 1)
        self.co_head = nn.Linear(84, 1)
        self.cu_head = nn.Linear(84, 1)
        self.fe2_head = nn.Linear(84, 1)
        self.fe_head = nn.Linear(84, 1)
        self.mg_head = nn.Linear(84, 1)
        self.mn_head = nn.Linear(84, 1)
        self.po4_head = nn.Linear(84, 1)
        self.so4_head = nn.Linear(84, 1)
        self.zn_head = nn.Linear(84, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # shape of x is (batch)x(2 * feature_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.cls_head(x)
        x = self.sigmoid(x)
        return x

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

            print(inputs)
            print(inputs.shape)
            print(labels)
            print(labels.shape)
            
            outputs = model(inputs)
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
                
                #wandb.log(x)
                
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

    # Load DataLoaders
    batch_size = config.batch_size
    shuffle = config.shuffle
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # Model and Optimizer
    model = ProteinClassifier()
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize wandb
    # wandb.login()
    # wandb.init(project=config.wandb_proj, name=config.run_name)

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
    
        #wandb.log(x)
    
        running_vloss = 0
        model.eval()
    
        val_metric.reset()
        
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                vlabels = vlabels.to(torch.int64)
                vaupr = val_metric(voutputs, vlabels)
    
                #wandb.log({'val/loss': vloss})
    
        avg_vloss = running_vloss / (i+1)
        vaupr = val_metric.compute()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('AUPR train {} valid {}'.format(torch.mean(aupr).item(), torch.mean(vaupr).item()))

        x = {'val/avg_loss': avg_vloss, 'epoch':epoch}
        for i, ion in enumerate(ions):
            x[f'val/aupr_{ion}'] = float(vaupr[i])
        #wandb.log(x)
    
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