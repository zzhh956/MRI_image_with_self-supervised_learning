import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import resnet18
        
class ResNet18(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_size = 512
        self.backbone = resnet18(pretrained = False, num_classes = self.embedding_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features = self.embedding_size, out_features = 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        embedding = self.backbone(x)
        features = self.mlp(embedding)
        
        return embedding, features

def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]

    return max(accs)
