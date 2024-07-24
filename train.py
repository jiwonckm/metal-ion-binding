import torch
import torchvision
from datetime import datetime
from dataloader import MionicDataset, MionicDatamodule, ConDatamodule
from model import TransformerEncoder, IonClassifier
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

# def train_one_epoch(epoch_index, device, model, train_loader, loss_fn, optimizer):
#     metric = MultilabelAveragePrecision(num_labels=len(ions), average=None, thresholds=None)
#     metric.to(device)
#     batch_metric = MultilabelAveragePrecision(num_labels=len(ions), average=None, thresholds=None)
#     batch_metric.to(device)

#     n = 0
#     avg_loss = 0
    
#     for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        
#         inputs, labels = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
        
#         outputs = model(inputs)
        
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         loss = loss.item()

#         b = len(inputs)
#         n += b
#         delta = b * (loss - avg_loss)
#         avg_loss += delta / n
        
#         labels = labels.to(torch.int64)
#         aupr = metric(outputs, labels)
        
#         if i%100 == 99:
#             print('   batch {} loss: {}'.format(i+1, avg_loss))
#             x = {'train/batch_loss': avg_loss}
#             wandb.log(x)
            
#     aupr = metric.compute()
#     return avg_loss, aupr

def main(config):
    
    # Set CUDA device
    if (config.device == 'cpu') or (not torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{config.device}')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load DataLoaders
    dataset = MionicDataset(config.data_dir, config.rep_dir, config.truth_dir)
    dm = MionicDatamodule(dataset, config.batch_size, config.shuffle, config.num_samples)
    dm.setup()
    # ion_weights = dm.setup()
    # ion_weights = ion_weights.to(device)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    cdm = ConDatamodule(dataset, config.batch_size, config.shuffle, config.con_num_samples)
    cdm.setup()
    contrastive_generator = cdm.train_dataloader()
    
    # Model and Optimizer
    cmodel = TransformerEncoder()
    cmodel = cmodel.to(device)
    closs_fn = MarginScheduledLossFunction()
    coptimizer = torch.optim.AdamW(cmodel.parameters(), lr=config.clr)

    model = IonClassifier()
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Initialize wandb
    wandb.login()
    wandb.init(project=config.wandb_proj, name=config.run_name, config=dict(config))

    # Metric
    metric = MultilabelAveragePrecision(num_labels=len(ions), average=None, thresholds=None)
    metric.to(device)
    batch_metric = MultilabelAveragePrecision(num_labels=len(ions), average=None, thresholds=None)
    batch_metric.to(device)
    val_metric = MultilabelAveragePrecision(num_labels=len(ions), average=None, thresholds=None)
    val_metric.to(device)

    # Run training
    EPOCHS = config.epochs
    best_vloss = 1_000_000
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch))

        # Contrastive Step
        if config.contrastive:
            cmodel.train(True)
            for i, batch in tqdm(enumerate(contrastive_generator), total=len(contrastive_generator)):
                pos1, pos2, neg = batch
                
                pos1 = cmodel(pos1.to(device))
                pos2 = cmodel(pos2.to(device))
                neg = cmodel(neg.to(device))
                
                contrastive_loss = closs_fn(pos1, pos2, neg)

                coptimizer.zero_grad()
                contrastive_loss.backward()
                coptimizer.step()
                
            closs_fn.step()
            cmodel.train(False)

        # Classifier Training Step
        model.train(True)
        n = 0
        avg_loss = 0
        
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            inputs, labels = data
            inputs = cmodel(inputs.to(device))
            labels = labels.to(device)
            
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
    
            b = len(inputs)
            n += b
            delta = b * (loss - avg_loss)
            avg_loss += delta / n
            
            labels = labels.to(torch.int64)
            aupr = metric(outputs, labels)
            
            if i%100 == 99:
                print('   batch {} loss: {}'.format(i+1, avg_loss))
                x = {'train/batch_loss': avg_loss}
                wandb.log(x)
                
        aupr = metric.compute()

        x = {'train/loss': avg_loss, 'epoch': epoch}
        for i, ion in enumerate(ions):
            x[f'train/aupr_{ion}'] = float(aupr[i])
        wandb.log(x)
        
        # Validation
        avg_vloss = 0
        model.eval()
        val_metric.reset()
        
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = data
                vinputs = cmodel(vinputs.to(device))
                vlabels = vlabels.to(device)
                
                voutputs = model(vinputs)

                vloss = loss_fn(voutputs, vlabels).item()
                vdelta = len(vinputs) * (vloss - avg_vloss)
                avg_vloss += vdelta / len(vinputs)
                
                vlabels = vlabels.to(torch.int64)
                
                vaupr = val_metric(voutputs, vlabels)
        
        vaupr = val_metric.compute()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('avg AUPR train {} valid {}'.format(torch.mean(aupr).item(), torch.mean(vaupr).item()))

        x = {'val/loss': avg_vloss, 'epoch': epoch}
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