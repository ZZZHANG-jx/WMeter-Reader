from math import log
import torch
import torch.nn as nn
from torch.nn import init
import functools
# from cbam import CBAM
import torch.nn.functional as F
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

class Lstm_Fc(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,embedding_dim=32,lstm_num=8,lstm_hiden_dim=64):
        super().__init__()
        self.embedding = torch.nn.Embedding(30,embedding_dim)
        self.embedding_fc = nn.Linear(embedding_dim,embedding_dim)

        # self.cnn = nn.Conv1d(embedding_dim,embedding_dim,kernel_size=3,stride=1,padding=1)

        self.lstm = nn.GRU(embedding_dim, lstm_hiden_dim,lstm_num,bidirectional=True,batch_first=True)
        
        self.classifier = nn.Linear(lstm_hiden_dim,lstm_hiden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hiden_dim*2,lstm_hiden_dim*2),
            nn.LeakyReLU(),
            nn.Linear(lstm_hiden_dim*2,lstm_hiden_dim*2),
            nn.LeakyReLU(),
            nn.Linear(lstm_hiden_dim*2,10),
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, 9, embedding_dim))

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_fc(x)  ## x->  batch, length, hidden_dim 
        x += self.pos_embedding
        x,hidden = self.lstm(x) ## x->  batch, length, hidden_dim 
        b,l,h = x.shape[:]
        # x = x.reshape(-1,h)
        x = self.classifier(x)
        return x