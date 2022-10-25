import os
import json
from typing import NamedTuple
from tqdm import tqdm
import torch
import torch.nn as nn

class Config(NamedTuple):
    #'Hyperparameter collections.namedtuple()'
    seed: int = 3431 # random seed
    batch_size: int = 64
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    momentum: float = 0.9
    weight_decay: float = 0.0001

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file,'r')))

class Trainer(object):
    def __init__(self, cfg, model, train_loader, val_loader, optimizer, schedule, save_dir, device):
        self.cfg=cfg
        self.model=model
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.optimizer=optimizer
        self.schedule=schedule
        self.save_dir=save_dir
        self.device=device
    
    def train(self, get_loss):
        self.model.train()
        model=self.model.to(self.device)

        global_step=0
        iter_bar=tqdm(self.train_loader, desc='Iter (loss=X.XXX)')
        for epoch in range(self.cfg.n_epochs):
            loss_sum=0
            for i, batch in enumerate(iter_bar):
                batch=[t.to(self.device) for t in batch]

                loss=get_loss(model, batch, global_step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_step+=1
                loss_sum+=loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)' %loss.item())
                
                if global_step%self.cfg.save_steps==0:
                    self.save(global_step)
                
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(epoch+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step)
                    return
            print('Epoch %d/%d : Average Loss %5.3f'%(epoch+1, self.cfg.n_epochs, loss_sum/(i+1)))
            self.schedule.step()
        self.save(global_step)
    
    def eval(self, evaluate):
        self.model.eval()
        model=self.model.to(self.device)
        results=[]
        iter_bar=tqdm(val_loader, desc='Iter (loss=X.XXX)')
        with torch.no_grad():
            for i, batch in enumerate(iter_bar):
                batch=[t.to(self.device) for t in batch]
                accuracy, result=evaluate(model, batch)
                results.append(result)
                iter_bar.set_description('Iter(acc=%5.3f)' %accuracy)
        return results
    
    def save(self, i):
        torch.save(self.model.state_dict(),         #model name and parameter
                    os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))
