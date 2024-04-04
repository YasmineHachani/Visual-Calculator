import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from argparse import ArgumentParser
from functools import partial

from Models.backbones2D.SimpleConv import SimpleConv
from Models.backbones2D.ResNet18 import ResNet18
from Models.backbones2D.ResNet34 import ResNet34
from Models.backbones2D.Vgg11 import Vgg11
from Models.backbones2D.Vit import Vit




# ------------
# Choosing model from ['SimpleConv', 'ResNet18', 'ResNet34', 'Vgg11','Vit']
# ------------

class ClassificationModel2D(pl.LightningModule) :
    """
        Parameters
        ----------
        model : str - Type of model model among ['SimpleConv', 'ResNet18', 'ResNet34', 'Vgg11','Vit']

        Returns
        -------
        ClassificationModel : pl.LightningModule
    """
    def __init__(self, model, criterion_name, optimizer, learning_rate, l2, schedule, NBClass, **kwargs) :
        super().__init__()
        self.model = self.init_model(model, NBClass, **kwargs)
        self.NBClass = NBClass
        self.criterion_name = criterion_name
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.l2 = l2
        self.schedule = schedule

    def init_model(self, model, NBClass, **kwargs) :
        if model == 'SimpleConv' :
            return SimpleConv(NBClass)
        elif model == 'ResNet18':
            return ResNet18(NBClass, **kwargs)
        elif model == 'ResNet34':
            return ResNet34(NBClass, **kwargs)
        elif model == 'Vgg11':
            return Vgg11(NBClass, **kwargs)
        elif model == 'Vit' :
            return Vit(NBClass, **kwargs)
        else :
            print(f'model {model} not available')
    
    def configure_optimizers(self):
        d = {}
        if self.optimizer == 'Adam':
            d['optimizer'] = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        elif self.optimizer == 'AdamW':
            d['optimizer'] = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        elif self.optimizer == 'RMSprop':
            d['optimizer'] = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.l2)
            print(f'Optimizer : {d["optimizer"]}')

        if self.schedule == 'StepLR' :
            d['lr_scheduler'] = { "scheduler": torch.optim.lr_scheduler.StepLR(d['optimizer'], step_size=30, gamma=0.1),
                               "interval": "epoch",
                               "frequency": 1}
        elif self.schedule == 'ReduceLROnPlateau':
            d['lr_scheduler'] = { "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(d['optimizer']),
                               "interval": "epoch",
                               "frequency": 1,
                               "monitor": "val/losses"}
        elif self.schedule == 'CyclicLR' :
           d['lr_scheduler'] = { "scheduler": torch.optim.lr_scheduler.CyclicLR(d['optimizer'],
                                                     base_lr=self.lr/100,
                                                     max_lr=self.lr,
                                                     cycle_momentum=False,
                                                     step_size_up=50, step_size_down=50),
                               "interval": "epoch",
                               "frequency": 1}
           # instantiate the WeakMethod in the lr scheduler object into the custom scale function attribute
           d['lr_scheduler']["scheduler"]._scale_fn_custom = d['lr_scheduler']["scheduler"]._scale_fn_ref()
           
           # remove the reference so there are no more WeakMethod references in the object
           d['lr_scheduler']["scheduler"]._scale_fn_ref = None
           
        elif self.schedule == 'WarmupDecay' :
            def warmupdecay(iteration, end_warmup, decay_step) :
                if iteration <= end_warmup :
                    return iteration/end_warmup
                else :
                    return (0.5)**((iteration - end_warmup)//decay_step)
            scheduler = partial(warmupdecay, end_warmup=20, decay_step=40)
            d['lr_scheduler'] = { "scheduler": torch.optim.lr_scheduler.LambdaLR(d['optimizer'], lr_lambda=scheduler),
                               "interval": "epoch",
                               "frequency": 1}
        return d


    def forward(self, batch) :
        """
        Produce a classification prediction using the model and the batch

        Parameters
        ----------
        batch : Dict containing
            'image' (NBImages, C, W, H)

        Returns
        -------
        batch['Pred'] : Tensor (NBImages, NBClass)
        """

        input = batch['image']
        batch['Pred'] = self.model(input.to(batch['label'].device)) # (Nimages, NBClass)
        return batch

    def prediction(self, image) :
        """
        Produce a classification prediction using the model and the batch
        """
        return self.model(image)

    def Criterion(self, batch) :
        """"
        Comput loss for the batch
        """
        
        if self.criterion_name == 'ce' :
            losses = nn.functional.cross_entropy(batch['Pred'], batch['label'], reduction='none')
        if self.criterion_name == 'ce_balanced' :
            class_weight = torch.tensor([1.21875,1.17672414,1.1375,1.625,1.1375,0.875,0.89802632,0.58836207]).to(batch['Class'].device)
            losses = nn.functional.cross_entropy(batch['Pred'], batch['label'], reduction='none', weight=class_weight)
        return {'losses' : losses}

    def Evaluations(self, batch, evals) :
        """"
        Args : batch with at least keys 'label'
        Update Evals : 'loss','accs','label', pred_i
        """
        evals['loss'] = evals['losses'].mean() # Loss is necessary for backprop in Pytorch Lightning
        batch['Pred'].argmax(axis=-1) #(Nvideos, 1)
        evals['preds'] =  batch['Pred'].argmax(axis=-1)
        evals['accs'] = (evals['preds'] == batch['label']).to(torch.float)
        evals['label'] =  batch['label']

        for i in range(self.NBClass) :
            evals[f'preds_{i}'] = (evals['preds'] == i).to(torch.float)

    def Logging(self, evals, step_label) :
        """
        pl logger for wandb
        """

        log = lambda k : self.log(f'{step_label}/{k}', evals[k].mean(), on_epoch=True, on_step=False)

        for k in ['losses', 'accs'] + [f'preds_{i}' for i in range(self.NBClass)]:
            evals[k] = evals[k].detach()
            log(k)

    def step(self, batch, step_label):
        batch = self.prediction(batch)
        evals = self.Criterion(batch)
        self.Evaluations(batch, evals)
        self.Logging(evals, step_label)
        return evals, batch

    def training_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'train')
        return evals

    def validation_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'val')
        return evals

    def test_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'test')
        return evals
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', '-bb', type=str, choices=['SimpleConv', 'ResNet18', 'ResNet34', 'Vgg11','Vit'], default='ResNet18')
        parser.add_argument('--pretrained_bb', '-pb', action='store_true', help='Use pretrained model')
        parser.add_argument('--criterion_name', type=str, choices=['ce', 'ce_balanced'], default='ce')
        parser.add_argument('--NBClass',type=int,default=13)
        parser.add_argument('--learning_rate','-lr',type=float, default=1e-4)
        parser.add_argument('--optimizer',type=str,choices=['Adam','SGD', 'AdamW'],default='Adam')
        parser.add_argument('--schedule',type=str, default='null')
        parser.add_argument('--l2',type=float, default=0.01)
        return parser
