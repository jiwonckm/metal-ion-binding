import torch
import torchvision
from dataloader import MionicDataset, MionicDatamodule, ConDatamodule
from model import IonBindingModel
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import MultilabelAveragePrecision
import wandb
from argparse import ArgumentParser
from omegaconf import OmegaConf
from margin import MarginScheduledLossFunction
import os

ions = ['Ca', 'Co', 'Cu', 'Fe2', 'Fe', 'Mg', 'Mn', 'PO4', 'SO4', 'Zn']
classes = ions + ['Null']

def main(config):
    
    # Set CUDA device
    if (config.device == 'cpu') or (not torch.cuda.is_available()):
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{config.device}')

    # Load DataLoaders
    train_dataset = MionicDataset(config.train_data_dir, config.rep_dir, config.truth_dir, config.num_samples)
    train_dm = MionicDatamodule(train_dataset, config.batch_size, config.shuffle)
    train_dm.setup()
    # ion_weights = dm.setup()
    # ion_weights = ion_weights.to(device)
    train_loader = train_dm.dataloader()

    val_dataset = MionicDataset(config.val_data_dir, config.rep_dir, config.truth_dir, config.val_num_samples)
    val_dm = MionicDatamodule(val_dataset, config.batch_size, config.shuffle)
    val_dm.setup()
    val_loader = val_dm.dataloader()
    
    cdm = ConDatamodule(train_dataset, config.batch_size, config.shuffle, config.con_num_samples)
    cdm.setup()
    con_generator = cdm.dataloader()
    
    # Model and Optimizer
    model = IonBindingModel()
    model = model.to(device)
    
    closs_fn = MarginScheduledLossFunction()
    coptimizer = torch.optim.AdamW(model.parameters(), lr=config.clr)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Initialize wandb
    wandb.login()
    wandb.init(project=config.wandb_proj, name=config.run_name, config=dict(config))

    # Create directory
    os.makedirs('{}/{}'.format(config.model_save_dir, config.run_name), exist_ok=True)

    # Metric
    metric = MultilabelAveragePrecision(num_labels=len(classes), average=None, thresholds=None)
    metric.to(device)
    batch_metric = MultilabelAveragePrecision(num_labels=len(classes), average=None, thresholds=None)
    batch_metric.to(device)
    val_metric = MultilabelAveragePrecision(num_labels=len(classes), average=None, thresholds=None)
    val_metric.to(device)

    # Run training
    EPOCHS = config.epochs
    best_vloss = 1_000_000
    best_vaupr = -1
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch))
        model.train(True)
        # Contrastive Step
        if config.contrastive:
            for i, batch in tqdm(enumerate(con_generator), total=len(con_generator)):
                pos1, pos2, neg = batch
                
                pos1 = model.project(pos1.to(device))
                pos2 = model.project(pos2.to(device))
                neg = model.project(neg.to(device))
                
                contrastive_loss = closs_fn(pos1, pos2, neg)

                coptimizer.zero_grad()
                contrastive_loss.backward()
                coptimizer.step()
                
            closs_fn.step()

        # Classifier Training Step
        n = 0
        avg_loss = 0
        
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            inputs, labels = data
            labels = labels.to(device)
            outputs = model(inputs.to(device))

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
        for i, ion in enumerate(classes):
            x[f'train/aupr_{ion}'] = float(aupr[i])
        wandb.log(x)
        
        # Validation
        avg_vloss = 0
        model.eval()
        val_metric.reset()
        
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_loader), total=len(val_loader)):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                
                voutputs = model(vinputs)

                vloss = loss_fn(voutputs, vlabels).item()
                vdelta = len(vinputs) * (vloss - avg_vloss)
                avg_vloss += vdelta / len(vinputs)
                
                vlabels = vlabels.to(torch.int64)
                
                vaupr = val_metric(voutputs, vlabels)
        
        vaupr = val_metric.compute()
        avg_vaupr = torch.mean(vaupr).item()
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('avg AUPR train {} valid {}'.format(torch.mean(aupr).item(), avg_vaupr))

        x = {'val/loss': avg_vloss, 'epoch': epoch}
        for i, ion in enumerate(classes):
            x[f'val/aupr_{ion}'] = float(vaupr[i])
        wandb.log(x)
    
        if avg_vaupr > best_vaupr:
            best_vaupr = avg_vaupr
            model_path = '{}/{}/epoch{}_{}'.format(config.model_save_dir, config.run_name, epoch, round(avg_vaupr,3))
            torch.save(model.state_dict(), model_path)
        elif epoch%5 == 0:
            model_path = '{}/{}/epoch{}_{}'.format(config.model_save_dir, config.run_name, epoch, round(avg_vaupr,3))
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