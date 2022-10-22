import os
import json
from typing import NamedTuple
from tqdm import tqdm
import torch
import torch.nn as nn
import checkpoint

class Config(NamedTuple):
    #'Hyperparameter collections.namedtuple()'
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train
    def from_json(cls, file):
        return cls(**json.load(open(file,'r')))

class Trainer(object):
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg=cfg
        self.model=model
        self.data_iter=data_iter
        self.optimizer=optimizer
        self.save_dir=save_dir
        self.device=device
    
    def train(self, get_loss):
        self.model.train()
        model=self.model.to(self.device)

        global_step=0
        iter_bar=tqdm(self.data_iter['train'], desc='Iter (loss=X.XXX)')
        for epoch in range(self.cfg.n_epoches):
            loss_sum=0
            for i, batch in enumerate(self.data_iter):
                batch=[t.to(self.device) for t in batch]
                loss=get_loss(model, batch, global_step)
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
        self.save(global_step)
    
    def eval(self, evaluate):
        self.model.eval()
        model=self.model.to(self.device)
        results=[]
        iter_bar=tqdm(self.data_iter['test'], desc='Iter (loss=X.XXX)')
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
