"""

LASER Finetuning for HuBERT

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import math
import os
import random
from pathlib import Path

import s3prl.hub as hub
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import LIBRISPEECH
import wandb
from torch.nn import functional as F
from .soft_dtw import SoftDTW
from .laser import IDMContrastiveLoss



## upstream dimension = 768 for HuBERT base
## upstream rate = 20 ms for HuBERT base

# intialize wandb with project name s3prl-dummy
wandb.init(project="Interspeech- 2024-CustomLoss")

class DownstreamExpert(nn.Module):

    def __init__(self, upstream_dim, upstream_rate, downstream_expert,
                 expdir, **kargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        ## modelrc is not used in this expert
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir

        self.train_dataset = LIBRISPEECH(root = self.datarc['path'], 
                                                url='train-clean-100', download=True)

        ## no connector needed for this expert
        # make connector as the two linear layers
        print(f"Upstream dim: {upstream_dim}")
        # self.connector = nn.Sequential(
        #     nn.Linear(upstream_dim, self.modelrc['input_dim']),
        #     nn.ReLU(),
        #     nn.Linear(self.modelrc['input_dim'], self.modelrc['input_dim'])
        # )

        self.connector = nn.Linear(upstream_dim, self.modelrc['input_dim'])

        self.loss_type = self.modelrc['loss_type']
        self.sigma = self.modelrc['sigma']
        self.margin = self.modelrc['margin']
        self.gamma = self.modelrc['gamma']   
        self.alpha = self.modelrc['alpha']

        self.objective1 = SoftDTW(gamma=self.gamma, normalize=True)
        self.objective2 = IDMContrastiveLoss(sigma=self.sigma, margin=self.margin)
        
        # mulipy the objective2 with alpha

        self.register_buffer("best_score", torch.tensor(float("inf")))

        # set config for wandb
        wandb.config.update(self.datarc)
        wandb.config.update(self.modelrc)


    
    # Interface
    def get_dataloader(self, split, epoch: int = 0):

        if split == "train":
            return self._get_train_dataloader(self.train_dataset, epoch)
        
    def _get_train_dataloader(self, dataset, epoch: int):
        from s3prl.utility.data import get_ddp_sampler
        sampler = get_ddp_sampler(dataset, epoch)

        return DataLoader(
            dataset, batch_size=self.datarc["train_batch_size"],
            shuffle = (sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn
            )

    # Interface
    def forward(self, split, features, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features1 = features[0].unsqueeze(0)
        features2 = features[1].unsqueeze(0)

        features1 = self.connector(features1)
        features2 = self.connector(features2)

        # l2 normalization
        features1 = F.normalize(features1, p=2, dim=2).to(device)
        features2 = F.normalize(features2, p=2, dim=2).to(device)


        loss1 = self.objective1(features1, features2)
        loss2 = self.objective2(features1) + self.objective2(features2)
            
        records["loss1"].append(loss1.item())
        records["loss2"].append(loss2.item())

        if self.loss_type == "softdtw":
            loss = loss1
        elif self.loss_type == "softdtw_lav":
            loss = loss1 + self.alpha*loss2

        records["loss"].append(loss.item())

        return loss

    
    # interface
    def log_records(self, split, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["loss", "loss1", "loss2"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'content_preserving-{split}/{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir)/ "log.log", 'a') as f:
                if key == 'loss' or key == 'loss1' or key == 'loss2':
                    print(f"\n{split} {key}: {average}")
                    f.write(f'\n{split} at step {global_step}: {average}\n')
                    wandb.log({f'{split}_{key}': average}, step=global_step)

                    if split == 'train' and key=='loss' and average < self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {split} at step {global_step}: {average}\n')
                        save_names.append(f'{split}-best.ckpt')
