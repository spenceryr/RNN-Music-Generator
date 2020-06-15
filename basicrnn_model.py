import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import torchvision
from torchvision import transforms, utils
import os

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           batch_first=True)
        
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, inp, hidden):
        out, hidden = self.rnn(inp, hidden)
        hidden = hidden.detach()
        return self.fc(out), hidden
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size).to('cuda' if next(self.parameters()).is_cuda else 'cpu')