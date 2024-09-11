import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel_im_distributionsrc(nn.Module):
    def __init__(self, intoken=31, outtoken=13, hidden=128, nlayers=3, dropout=0.1):
        super(TransformerModel_im_distributionsrc, self).__init__()
        # nhead = hidden // 64
        nhead = 4 
        self.fc_out1 = nn.Linear(512,hidden)
        self.distribution_embedding = nn.Linear(30,hidden)      ## ours
        # self.distribution_embedding = nn.Embedding(30,hidden) ## ours 离散textual feature

        self.encoder_x = nn.Embedding(200, hidden)
        self.encoder_y = nn.Embedding(200, hidden)

        self.modal_embedding = nn.Embedding(2, hidden)

        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.inscale = math.sqrt(intoken)
        self.outscale = math.sqrt(outtoken)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=nlayers,num_decoder_layers=nlayers, dim_feedforward=hidden, dropout=dropout)
        self.fc_out = nn.Linear(hidden,outtoken)
        # self.fc_out = nn.Sequential(
            # nn.Linear(hidden, hidden),
            # nn.BatchNorm1d(hidden),
            # nn.LeakyReLU(),
            # nn.Linear(hidden, int(hidden/2)),
            # nn.BatchNorm1d(int(hidden/2)),
            # nn.LeakyReLU(),
            # nn.Linear(int(hidden/2), outtoken),
            # )

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src,distribution, trg, im_embedding,src_x=0,src_y=0):


    # 用于模型参数量计算
    # def forward(self, im_embedding): # src (12,1) distribution(1,12,30),pred_temp(1,1), im_embedding(12,512)
        # src = torch.ones(12,1).cuda().long()*3
        # trg = torch.ones(1,1).cuda().long()*3
        # distribution = torch.ones(1,12,30).cuda()
        # im_embedding = torch.ones(12,512).cuda()

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)
        src_pad_mask_cat = torch.cat((src_pad_mask,src_pad_mask),dim=1) # ours
        # src_pad_mask_cat = src_pad_mask  # ours_without_visual/text
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        l,b = src.shape[:2]
        im_embedding = self.fc_out1(im_embedding)
        im_embedding = im_embedding.view(b,l,128).permute(1,0,2)
        im_embedding_position = self.pos_encoder(im_embedding)

        # src = self.encoder(src)
        src = self.distribution_embedding(distribution.permute(1,0,2))   ## ours tim sota
        # src = self.distribution_embedding(torch.argmax(distribution.permute(1,0,2),-1))   ## ours 离散textual feature
        src_position = self.pos_encoder(src)

        visual = self.modal_embedding(torch.ones(l,b).long().cuda())
        text = self.modal_embedding(torch.zeros(l,b).long().cuda())
        src_position_im = torch.cat((src_position+text,im_embedding_position+visual),dim=0) # ours
        # src_position_im = src_position+text # ours_without_visual
        # src_position_im = im_embedding_position+visual # ours_without_text

        # src_x = self.encoder_x(src_x)
        # src_y = self.encoder_y(src_y)
        
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)
        
        output = self.transformer(src_position_im, trg, tgt_mask=self.trg_mask,
                                  src_key_padding_mask=src_pad_mask_cat, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask_cat)

        l,b,dim = output.shape[:]
        output = self.fc_out(output.view(-1,dim)).view(l,b,13)

        return output