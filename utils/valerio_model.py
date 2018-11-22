import math, random, gym, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import utils.motion_data as mmd
from torch.autograd import Variable

class LSTM_FC_classifier(nn.Module):
    def __init__(self, num_inputs, lstm_hidden_size, batch_size, seq_length, num_outputs = 1):
        super(LSTM_FC_classifier, self).__init__()
        self.num_inputs   = num_inputs
        self.num_outputs  = num_outputs
        self.lstm_hidden_size = lstm_hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size = num_inputs, hidden_size = lstm_hidden_size, 
            num_layers = 1, batch_first = False, dropout = 0, bidirectional = False)
        '''
        LSTM:
        **output** of shape (seq_len, batch, hidden_size * num_directions): 
            tensor containing the output features `(h_t)` from the last layer of the LSTM, for each t. 
            If `torch.nn.utils.rnn.PackedSequence` was given as input, output will also be a packed sequence.
        **h_n** of shape (num_layers * num_directions, batch, hidden_size): 
            tensor containing the hidden state for `t = seq_len`
        **c_n** of shape (num_layers * num_directions, batch, hidden_size): 
            tensor containing the cell state for `t = seq_len`
        '''
        self.fc1 = nn.Linear(lstm_hidden_size, 16)
        self.fc2 = nn.Linear(16, 5)
        self.fc3 = nn.Linear(5, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLu()
        
    def forward(self, act_info):
        if act_info is None: #Train mode
            hx = Variable(torch.zeros(1, self.batch_size, self.lstm_hidden_size))
            cx = Variable(torch.zeros(1, self.batch_size, self.lstm_hidden_size))
        else: #Acting mode
            hx, cx = act_info
            inputs = act_info.view(1, 1, -1)

        lstm_outputs, (hx, cx) = self.lstm(inputs, (hx, cx))
        y = self.relu(self.fc1(lstm_outputs))
        y = self.relu(self.fc2(y))
        class_estimate = self.sigmoid(self.fc3(y))
        return class_estimate, (hx, cx)