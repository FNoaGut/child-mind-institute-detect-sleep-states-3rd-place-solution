import numpy as np
from torch import nn
import torch
import random

class Net(nn.Module):
    def __init__(
            self, cfg):
        super(Net, self).__init__()

        self.col_index = [cfg.all_features[i] for i in cfg.num_features]
        self.col_minute = [cfg.all_features[i] for i in cfg.col_minute]

        self.numerical_linear = nn.Sequential(
            nn.Linear(len(self.col_index), cfg.numeraical_linear_size),
        )

        self.rnn = nn.GRU(cfg.numeraical_linear_size, cfg.model_size,
                          num_layers=cfg.num_layers_gru,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.01)

        self.linear_out = nn.Sequential(
            nn.Linear(cfg.model_size * 2 + len(self.col_minute),
                      cfg.linear_out),
            )

        self.last_layer =  nn.Linear(cfg.linear_out,
                      cfg.out_size)

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """

        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

    def forward(self, numerical_array,
                mask_array,
                attention_mask,train=True):

        input_tensor = numerical_array[:, :, self.col_index]
        numerical_embedding = self.numerical_linear(input_tensor)
        output, _ = self.rnn(numerical_embedding)
        output = self.linear_out(torch.cat((output, numerical_array[:, :, self.col_minute]), dim=2))
        output = self.last_layer(output)

        return output
