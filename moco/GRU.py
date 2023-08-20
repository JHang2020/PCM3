from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn

from torch import optim
import torch.nn.functional as F

import numpy as np
import math
from torch.utils.data import random_split
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.num_layers = num_layers

    def forward(self, input_tensor, seq_len,all_seq=False):
        
        self.gru.flatten_parameters()

        encoder_hidden = torch.Tensor().to(device)
        
        for it in range(max(seq_len)):
          if it == 0:
            enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it+1, :])
          else:
            enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it+1, :], hidden_tmp)
          encoder_hidden = torch.cat((encoder_hidden, enout_tmp),1)#N T C
        if all_seq:
            return encoder_hidden
        
        hidden = torch.empty((1, len(seq_len), encoder_hidden.shape[-1])).to(device) #1 N C
        count = 0
        for ith_len in seq_len:
            hidden[0, count, :] = encoder_hidden[count, ith_len - 1, :]
            count += 1
        
        return hidden


class BIGRU(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, en_num_layers=3, num_class=60):
        super(BIGRU, self).__init__()
        self.en_num_layers = en_num_layers
        self.encoder = EncoderRNN(en_input_size, en_hidden_size, en_num_layers).to(device)
        self.fc = nn.Linear(2*en_hidden_size,num_class)

        self.input_norm = nn.BatchNorm1d(en_input_size)  #

        self.en_input_size = en_input_size
        #self.predictor = nn.Identity()

    def forward(self, input_tensor, knn_eval=False,all_seq=False,small_prompt=None,st_idx=None):

        input_tensor = self.input_norm(input_tensor.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()  # BN

        seq_len = torch.zeros(input_tensor.size(0),dtype=int) + input_tensor.size(1) #  list of input sequence lengths .

        encoder_hidden = self.encoder(
            input_tensor, seq_len,all_seq=all_seq)
        if knn_eval: # return last layer features during  KNN evaluation (action retrieval)
             return encoder_hidden[0], self.fc(encoder_hidden[0])
        elif all_seq:
            return encoder_hidden
        else:
            if small_prompt != None:
                prompt_size = small_prompt.shape[1]//2
                encoder_hidden[0][:,st_idx+1024:st_idx+1024+prompt_size] += small_prompt[:, prompt_size:]
                encoder_hidden[0][:,st_idx:st_idx+prompt_size] += small_prompt[:, 0:prompt_size]
            out = self.fc(encoder_hidden[0])
            return out

class BIGRU_part(nn.Module):
    def __init__(self, en_input_size, en_hidden_size, en_num_layers=3, num_class=60):
        super(BIGRU_part, self).__init__()
        self.en_num_layers = en_num_layers
        self.encoder = EncoderRNN(en_input_size, en_hidden_size, en_num_layers).to(device)
        self.fc = nn.Linear(2*en_hidden_size-512,num_class)

        self.input_norm = nn.BatchNorm1d(en_input_size)  #

        self.en_input_size = en_input_size
        #self.predictor = nn.Identity()

    def forward(self, input_tensor, knn_eval=False):

        input_tensor = self.input_norm(input_tensor.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()  # BN

        seq_len = torch.zeros(input_tensor.size(0),dtype=int) + input_tensor.size(1) #  list of input sequence lengths .

        encoder_hidden = self.encoder(
            input_tensor, seq_len)
        if knn_eval: # return last layer features during  KNN evaluation (action retrieval)
             return encoder_hidden[0][:,-512:], self.fc(encoder_hidden[0][:,:-512])
        else:
             out = self.fc(encoder_hidden[0][:,:-512])
             return out