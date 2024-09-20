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
    test_dataset = MionicDataset(config.test_data_dir, config.test_rep_dir, config.truth_dir, 1000)
    test_dm = MionicDatamodule(test_dataset, config.batch_size, shuffle=False)
    test_loader = test_dm.dataloader()

    # Load model
    model = IonBindingModel()
    model = model.to(device)
    model.load_state_dict(torch.load(config.model_dir))
    model.eval()

    # Metric
    metric = MultilabelAveragePrecision(num_labels=len(classes), average=None, thresholds=None)
    metric.to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.int64)
            
            outputs = model(inputs)
            
            aupr = metric(outputs, labels)
    
    aupr = metric.compute()
    avg_aupr = torch.mean(aupr).item()
    
    with open(f"test_results/{config.file_name}.txt", "w") as file:
        file.write('Test AUPR for each ion:\n')
        for i, ion in enumerate(classes):
            file.write(f'{ion}: {float(aupr[i])}\n')
        file.write(f'\nAvg AUPR: {avg_aupr}')

if __name__ == "__main__":
    parser = ArgumentParser(description='Test')
    parser.add_argument('--model_dir', type=str, help='Directory of model')
    parser.add_argument('--file_name', type=str, help='Directory of results file')
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--device', required=True, help='Specify device')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.model_dir = args.model_dir
    config.file_name = args.file_name
    config.device = args.device
    
    main(config)