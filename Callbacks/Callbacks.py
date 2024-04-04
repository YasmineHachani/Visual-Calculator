import pytorch_lightning as pl
import flowiz
import torch
import wandb
from ipdb import set_trace
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------
# Manual result logging
# ouput: results.csv file containing all epochs prediction on test and validation and the test prediction
#        results_summary.csv containing summary evaluation of test
# ------------

class ResultsLogger(pl.Callback) :
    def __init__(self, keys=['losses', 'accs', 'preds', 'label'], filepath=None):
        super().__init__()
        self.fp = filepath
        self.keys = keys

    def setup(self, trainer, pl_module, stage) :
        if self.fp is None :
            self.fp = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.experiment.id, 'results.csv')
        print(f'Save results in {self.fp}')
        with open(self.fp, 'w') as f :
            f.write(f'epoch,step_label,file_name,'+','.join(self.keys)+'\n')

    @torch.no_grad()
    def write_results(self, imps, outputs, epoch, step_label) :
        with open(self.fp, 'a') as f :
            for i, imn in enumerate(imps) :

                f.write(f'{epoch},{step_label},{imn.strip(os.environ["PWD"])},'+','.join([f'{outputs[j][i].item():.3f}' for j in self.keys if j in outputs.keys()])+'\n')

    def batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, step_label):
           key_path = "image_name"
           self.write_results(batch[key_path], outputs, trainer.current_epoch, step_label)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'test')

    def on_test_end(self, trainer, pl_module) :
        self.summary_path = self.fp.replace('.csv', '_summary.csv')
        dfr = pd.read_csv(self.fp)
        mean_keys = [key for key in self.keys if 'accs' in key or 'losses' in key]
        dfr.groupby(['step_label','epoch']).mean()[mean_keys].to_csv(self.summary_path, sep='\t')
        print(f'Summary saved at : {self.summary_path}')

#TO DO : add min/max mode and min_delta
class CustomEarlyStopping(pl.Callback):
    def __init__(self, monitor_metric='val/losses', patience=3, epoch_window_size=5):
        super().__init__()
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.epoch_window_size = epoch_window_size
        self.counter = 0
        self.metrics_history = []
        self.best_average_metric = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        current_metric = trainer.callback_metrics.get(self.monitor_metric)

        if current_metric is None:
            raise ValueError(f'Metric {self.monitor_metric} not found in callback_metrics. Make sure it is being logged.')

        self.metrics_history.append(current_metric)

        if len(self.metrics_history) == self.epoch_window_size:
            
            current_average_metric = sum(self.metrics_history) / len(self.metrics_history)

            if current_average_metric <= self.best_average_metric:
                self.best_average_metric = current_average_metric
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                print(f"Early stopping triggered. No improvement in average {self.monitor_metric} for {self.patience} consecutive epochs.")
                trainer.should_stop = True

            self.metrics_history = []
