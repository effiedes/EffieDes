import torch

from .CbNet_opti import *
from .CbNet_optiR import *

class Load_model(nn.Module):
    def __init__(
        self,
        version="v3",
        multi = True,
        tm = False,
        unary = False
    ):
        super(Load_model, self).__init__()
        self.version = version
        self.multi = multi
        self.tm = tm
        self.unary = unary


    def forward(self, model_name=None, model_path="../Results/model/", device=torch.device("cpu")):
    
        """
        Load a trained model. If model_name = None, a randomly init net is returned.
        """
        
        nb_aa = 20
        # Parameters for gMLP
        seq_len = 48
        gMLP_output_dim = 32*2*2
        # Parameters for ResMLP
        hidden = 128*2
        block_size = 2

        if self.version == "v2":
            # Parameters for gMLP
            embed_dim = 128
            depth_gMLP = 12+3
            ff_mult = 4
            # Parameters for ResMLP
            nblocks=10
            
            model = CbNet_v2(embed_dim=embed_dim,
                depth_gMLP=depth_gMLP,
                ff_mult=ff_mult,
                seq_len = seq_len,
                gMLP_output_dim = gMLP_output_dim,
                # Options for ResNet
                hidden=hidden,
                nblocks=nblocks,
                block_size=block_size,
                nb_aa=nb_aa,
                multichain=self.multi)
        
        if self.version == "v3":
            # Parameters for gMLP
            embed_dim = 128*2
            depth_gMLP = 4
            ff_mult = 2
            nb_it = 6
            # Parameters for ResMLP
            nblocks = 3
        
            model = CbNet_v3(embed_dim=embed_dim,
                depth_gMLP=depth_gMLP,
                ff_mult=ff_mult,
                seq_len = seq_len,
                gMLP_output_dim = gMLP_output_dim,
                # Options for ResNet
                hidden=hidden,
                nblocks=nblocks,
                block_size=block_size,
                nb_aa=nb_aa,
                nb_it=nb_it,
                multichain=self.multi,
                transmembrane=self.tm,
                unary=self.unary)
                
        model.to(device)
        
        if model_name is not None:
            checkpoint = torch.load(model_path+ model_name, map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])

        return model
