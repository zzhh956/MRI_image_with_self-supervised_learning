import ipdb
import torch
import random
import numpy as np
import pytorch_lightning as pl
from loss import xt_xent
from torch.optim import Adam
from models import ResNet18, KNN
from dataloader import ImageLoader
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class Image_self_supervise(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_total = 0
        self.train_loss = 0

        # model
        self.batch_size = 512
        self.learning_rate = 1e-4
        self.model = ResNet18()

    def prepare_data(self):
        self.dataset = ImageLoader('train')
        self.test_dataset = ImageLoader('test')

        self.dataset_len = len(self.dataset)
        self.train_idx = list(range(self.dataset_len))
        random.shuffle(self.train_idx)

    def train_dataloader(self):
        sampler = SubsetRandomSampler(self.train_idx)
        return DataLoader(self.dataset, batch_size = self.batch_size, sampler = sampler, num_workers = 8, pin_memory = True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = 8, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size = 1, num_workers = 8, pin_memory = True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr = self.learning_rate, weight_decay = 1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_dataloader()), eta_min = 0, last_epoch = -1)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        x1, x2 = self.dataset.Augment(batch['image'])

        _, y1 = self.model(x1)
        _, y2 = self.model(x2)
        
        loss = xt_xent(y1, y2)

        self.train_loss += loss.item()
        self.train_total += 1
        self.log('avg_loss', self.train_loss/self.train_total, prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        self.train_loss = 0
        self.train_total = 0

    def validation_step(self, batch, batch_idx):
        emb_y, _ = self.model(batch['image'])
        label = batch['label']
        
        emb_y = emb_y.reshape(-1, 512)
        label = label.reshape(-1)

        acc = KNN(emb_y, label, batch_size = batch['image'].size(0))

        return acc

    def validation_epoch_end(self, outputs):
        self.log('val_acc', min(outputs), prog_bar = True)
    
    def test_step(self, batch, batch_idx):
        emb, _ = self.model(batch['image'])

        return emb

    def test_epoch_end(self, outputs):
        emb_list = []

        for output in outputs:
            emb_list.append(output.cpu().numpy())

        emb_list = np.array(emb_list).reshape(-1, 512)

        np.save('../311513015', emb_list)

