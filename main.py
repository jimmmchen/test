import fire
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torchvision
import torchvision.transforms as transforms
#from torch.utils import set_seeds

import model.resnet as resnet
import train

def load_file(path):
    return np.load(path).astype(np.float32)

def get_model(model_name, device):
    if model_name=='Resnet':
        return resnet.ResNet(resnet.ResidualBlock, [3, 4, 6, 3]).to(device)

def main(task='Hemorrhage',
        train_cfg='config/train_Hemorrhage.json',
        model_name='Resnet',
        train_file="datasets/Hemorrhage/train/",
        test_file="datasets/Hemorrhage/val/",
        save_dir='save_model/resnet',
        mode='train'):

        cfg=train.Config.from_json(train_cfg)
        #model_cfg=resnet.Config.from_json(model_cfg)

        #set_seeds(cfg.seed)
        
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.49, 0.248),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.49, 0.248)
        ])
        train_dataset = torchvision.datasets.DatasetFolder(train_file, loader=load_file, extensions="npy", transform=train_transforms)
        val_dataset = torchvision.datasets.DatasetFolder(test_file, loader=load_file, extensions="npy", transform=val_transforms)
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        criterion=nn.CrossEntropyLoss()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=get_model(model_name, device)
        optimizer=torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        schedule=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6, last_epoch=-1)
        trainer = train.Trainer(cfg, 
                                model,
                                train_loader,
                                val_loader,
                                optimizer,
                                schedule,
                                save_dir, device)

        if mode=='train':
            def get_loss(model, batch, global_step):
                images, labels=batch
                output=model(images)
                loss=criterion(output, labels).to(device)
                return loss
            
            trainer.train(get_loss)
            
            def evalute(model, batch):
                images, labels=batch
                outputs=model(images)
                _, prediction=torch.max(outputs.data, 1)
                result=(prediction == labels).float()
                accuracy=result.mean()
                return accuracy, result
            
            results=trainer.eval(evaluate)
            total_accuracy=torch.cat(results).mean().item()
            print("Accuracy:", total_accuracy)

if __name__=='__main__':
    fire.Fire(main)
        
