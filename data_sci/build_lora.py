## https://github.com/alexriggio/BERT-LoRA-TensorRT/blob/main/src/lora_from_scratch.py
import math
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