
import torch
import torch.nn as nn
from .gMLP import *


def weights_init(m):
    """
    For initializing weights of linear layers (bias are put to 0).
    """

    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class MLP(nn.Module):

    """
    Define a MLP with layer norm & ReLU 
    Init: size of the input (int)
          size of the output (int)
          list of the hidden layer dimensions (there will be as many hidden layers as elements in the list)
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()

        self.MLP = torch.nn.Sequential()
        layer_sizes = [input_size]+hidden_sizes
        
        for k in range (1, len(layer_sizes)):
            self.MLP.add_module("Linear layer " + str(k), nn.Linear(layer_sizes[k-1], layer_sizes[k])),
            self.MLP.add_module("LN " + str(k), nn.LayerNorm(layer_sizes[k])),
            self.MLP.add_module("ReLU " + str(k), nn.ReLU())
        self.MLP.add_module("Output layer", nn.Linear(layer_sizes[-1], output_size))
            
        self.MLP.apply(weights_init)

    def forward(self, x):

        return self.MLP(x)
        
        
class ResBlock(nn.Module):

    """
    Residual block of 2 hidden layer for resMLP
    Init: size of the input (the output layer as the same dimension for the sum)
          size of the 1st hidden layer
          size of residual blocks (int, min and default: 2)
    """

    # In ResNet v2: BN, relu, weight, BN, relu, weight, sum
    # no dropout and kaiming init

    def __init__(self, input_size, hidden_size, block_size=2, BatchNorm=True):
        super(ResBlock, self).__init__()

        activation = torch.nn.GELU() 
        ### Input block ###
        self.block = torch.nn.Sequential()
        self.block.add_module("In_BN", nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
        #self.block.add_module("In_BN", nn.LayerNorm(input_size))
        self.block.add_module("relu_in", activation)
        self.block.add_module("In_layer", nn.Linear(input_size, hidden_size))

        ### Intermediate blocks (optionnal) ###
        for k in range(block_size - 2):
            self.block.add_module("BN_" + str(k), nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
            self.block.add_module("relu_" + str(k), activation)
            self.block.add_module("layer" + str(k), nn.Linear(hidden_size, hidden_size))

        ### Output block ###
        self.block.add_module("Out_BN", nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
        self.block.add_module("relu_out", activation)
        self.block.add_module("Out_layer", nn.Linear(hidden_size, input_size))

    def forward(self, x):

        x_out = self.block(x)
        x = x_out + x

        return x


class ResMLP(nn.Module):

    """
    MLP with residual connections.
    Init: number of blocks (nblocks, int)
          number of layer in a block (block_size, int)
          size of the output (output_size, int)
          size of the input (int)
          size of the hidden layers (hidden_size, int)
    """

    def __init__(self, output_size, input_size, hidden_size, nblocks=2, block_size=2, BatchNorm=True):
        super(ResMLP, self).__init__()

        self.ResNet = torch.nn.Sequential()
        self.ResNet.add_module("In_layer", nn.Linear(input_size, hidden_size))
        # self.ResNet.add_module("relu_1", torch.nn.ReLU())
        for k in range(nblocks):
            self.ResNet.add_module("ResBlock" + str(k), ResBlock(hidden_size, hidden_size, block_size, 
                                                                 BatchNorm=BatchNorm))
        self.ResNet.add_module("Final_BN", nn.BatchNorm1d(num_features=hidden_size, 
                                                          track_running_stats=False)
                              if BatchNorm else nn.LayerNorm(hidden_size))
        self.ResNet.add_module("relu_n", torch.nn.GELU())
        self.ResNet.add_module("Out_layer", nn.Linear(hidden_size, output_size))

    def forward(self, x):

        x = self.ResNet(x)

        return x


class ResNet(nn.Module):

    """
    Network composed of embedding + MLP
    Init: grid_size (int)
          hiddensize (int): number of neurons in hidden layer (suggestion: 128 or 256)
          resNet (bool). If False (default), use a regular MLP. Else, use a ResMLP
          nblocks (int): number of residual blocks. Default is 2
    """

    def __init__(self, input_size, nb_aa, hidden_size=128, resnet=True, nblocks=2, block_size=2,
                 BatchNorm=True):
        super(ResNet, self).__init__()

        # dropout_rate = 0.1
        self.nb_aa = nb_aa

        # ResMLP
        if resnet:
            self.MLP = ResMLP(self.nb_aa ** 2, input_size, hidden_size, nblocks, block_size, BatchNorm)

        # else, regular MLP
        else:
            self.MLP = MLP(self.nb_aa ** 2, input_size, [hidden_size])

        self.MLP.apply(weights_init)

    def forward(self, x, device, thresh=None, d=None):

        bs = 1
        nb_pairs, _ = x.shape
        nb_var = int(nb_pairs ** 0.5)  # len of the protein
        x = x.reshape(nb_pairs * bs, -1)  # bs*nb_pairs, nb_ft

        W = torch.zeros((nb_pairs, self.nb_aa ** 2), device=device, dtype=x.dtype)
        idx = torch.triu(torch.ones((nb_var, nb_var)), 1)
        idx = torch.flatten(idx.type(torch.BoolTensor)).to(device)

        if isinstance(thresh, int):
            idx *= torch.all((d < thresh).view(-1, 1), axis=1)  # line where d<thresh

        idx *= torch.all((d>0).view(-1, 1), axis = 1) #to not take missing residues into account

        pred = self.MLP(x[idx])
        W[idx] = pred
        W = W.reshape(bs, nb_var, nb_var, -1)
        W += W.transpose(1, 2).view(bs, nb_var, nb_var, self.nb_aa, self.nb_aa).transpose(3, 4).reshape(bs, nb_var, nb_var, -1)

        return W
