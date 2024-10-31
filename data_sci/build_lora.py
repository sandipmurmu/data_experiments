## https://github.com/alexriggio/BERT-LoRA-TensorRT/blob/main/src/lora_from_scratch.py
import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F




class LineraLoRA(nn.Module):
    """
    Arguments:
        in_dim      : an integer representing input dimesion
        out_dim     ; an integer representing output dimesion
        r           : rank of low rank approximated matrices
        lora_alpha  : represent the numerator of the scaling constant alpha/r
        lora_dropout: a float between 0 and 1, represents dropout probability
    """
    def __init__(self, in_dim: int, out_dim: int, r :int, lora_alpha: float, lora_dropout: float):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)

        # creates linear layer and freeze it
        self.pretrained = nn.Linear(in_dim, out_dim,bias=False)
        self.pretrained.weight.requires_grad = False

        # create a low rank A matrix and initialize the same as PEFT
        self.lora_A = nn.Linear(in_dim,r, bias=False)
        # kaiming normalizatio is a weight normalization method that adjust weights of NN layers to facilitate efficient
        # by addressing the vanishing or exploding gradient problem
        nn.init.kaiming_normal_(self.lora_A.weight, a=math.sqrt(5))

        # create a low rank B matrix and initialize to zero
        self.lora_B = nn.Linear(r,out_dim, bias=False)
        nn.init.constant_(self.lora_B.weight,0)

        self.scaling = self.lora_alpha/r

    
    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling
        return pretrained_out + lora_out
    
    def freeez_model(model: nn.Module):
        for name, param in model.named_parameters():
            if 'lora' not in name and 'classifier' not in name:
                param.requires_grad = False


    def unfreeze_model(model: nn.Module):
        for name, param in model.named_parameters():
            param.requires_grad = True

    
    def create_lora(module, r, lora_dropout, lora_alpha):
        """ convert  linear module to LoRA """
        k,d = module.weight.shape # linear weights are transposed, hence shape is (k,d) and (k,d)
        lora = LineraLoRA(d, k, r=r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
        with torch.no_grad():
            lora.pretrained.weight.copy_(module.weight)
            lora.pretrained.bias.copy_(module.bias)
        
        return lora
    
    def create_linear(module):
        """ convert lora to linear model"""
        k,d = module.pretrained.weight.shape
        linear = nn.Linear(d, k, bias=True)
        with torch.no_grad():
            linear.weight.copy_(module.pretrained.weight + (module.lora_A.weight @ module.lora_B.weight) * module.scaling)
            linear.bias.copy_(module.pretrained.bias)

        return linear
    
    def add_lora_layers(model, module_names: Tuple=("query", "value"), 
                        r: int=8, lora_alpa: float=16, lora_dropout: float=0.1, 
                        ignore_layers: List[int]=[]):
        
        module_types: Tuple=(nn.Linear,)

        # disable dropout in frozen layers
        for module in module.modules():
            if isinstance(module, nn.Dropout):
                module.p=0.0
        # replace chosen linear modules with lora modules
        for name, module in model.named_children():
            if isinstance(module, module_types) and name in module_names:
                temp_lora = create_lora(module,r=r,lora_dropout=lora_dropout, lora_alpha=lora_alpa)
                setattr(model, name, temp_lora)
            else:
                ignore_layers_str = [str(i) for i in ignore_layers]
                if name not in ignore_layers_str:
                    add_lora_layers(module, module_names, r, lora_dropout, lora_alpa, ignore_layers)

    
    
    def merge_lora_layers(model, module_names: Tupe=("query", "value"), dropout=0.1):
        """ replaces lora modules with original linear equivalents"""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        # replace linear modules with lora
        for name, module in model.named_children():
            if name in module_names and hasattr(module, "pretrained"):
                temp_linear = create_linear(module)
                setattr(model, name, temp_linear)
            else:
                merge_lora_layers(module, module_names=module_names, dropout=0.1)




    