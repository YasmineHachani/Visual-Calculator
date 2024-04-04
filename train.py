import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
import collections
import argparse
from pathlib import Path
import torch.nn as nn

from DataLoading.CachingDataLoader import CachingDataModule
from Models.models2D.ClassificationModel import ClassificationModel2D
from Callbacks.Callbacks import ResultsLogger, CustomEarlyStopping


#Handle arguments
parser = argparse.ArgumentParser()

parser = pl.Trainer.add_argparse_args(parser)
parser = CachingDataModule.add_specific_args(parser)
parser = ClassificationModel2D.add_specific_args(parser)
parser.add_argument("--exp_name", type=str, default='Test')
parser.add_argument("--project_name", default='VisualCalculator')
parser.add_argument('--seed', default=1405, type=int)
args = parser.parse_args()

args.base_dir = os.environ['PWD']
args.gpus = -1 if torch.cuda.is_available() else 0

#Seed
pl.seed_everything(args.seed)

#Init the model
model = ClassificationModel2D(**vars(args))

dm = CachingDataModule(**vars(args))

# ------------
# logger and callbacks
# ------------

args.save_dir = os.path.join(os.environ['PWD'], 'results/')
args.logger = WandbLogger(project=args.project_name,
                          name=args.exp_name,
                          save_dir=args.save_dir,
                          log_model=False)
model.hparams.project_name = args.logger.experiment.project_name()
model.hparams.experiment_id =  args.logger.experiment.id
model.hparams.experiment_name =  args.logger.name
wandb.run.log_code(".")


# ------------
# log model
# ------------
path_save_model = os.path.join(args.save_dir, args.logger.name, args.logger.experiment.id)
Path(path_save_model+'/checkpoints/').mkdir(parents=True, exist_ok=True)
mck = pl.callbacks.ModelCheckpoint(path_save_model+'/checkpoints/',
                                   monitor='val/losses',
                                   filename='epoch_{epoch}_val_accs_{val/losses:.3f}',
                                   save_last=True,
                                   mode='min',
                                   save_top_k=3
                                   #every_n_epochs=15,
                                   )

# ------------
# Hyper parameters
# ------------
early_stop_callback = [CustomEarlyStopping(monitor_metric="val/losses", patience=5, epoch_window_size=5)]
args.callbacks = [ResultsLogger(), mck] + early_stop_callback
args.max_epochs = 250
trainer = pl.Trainer.from_argparse_args(args)
trainer.logger.log_hyperparams(args)

"""
Learning rate finder :

import matplotlib.pyplot as plt
trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=10, accumulate_grad_batches=4)

lr_finder = trainer.tuner.lr_find(model, datamodule=dm, early_stop_threshold=None, min_lr=1e-4, num_training=100)
new_lr = lr_finder.suggestion()
print("Ideal lr : ", new_lr)

# Plot with
plt.xscale("log")
fig = lr_finder.plot(suggest=True)
fig.savefig("minmaxnorm643d_newnorm2.png")
"""
trainer.fit(model,dm)
trainer.test(model,dm)