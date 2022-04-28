import torch.nn as nn
import torch

from models.embed import Embedder
from models.decoder_layer import DecoderLayer,Norm
from models.utils import get_clones

class ImageFeatureDecoder(nn.Module):
    
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_length=1024):
        super().__init__()
        self.N = N
        self.embedX = Embedder(vocab_size, d_model)
        self.embedP = Embedder(max_length, d_model)
    # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        
    def forward(self,image_tensor,operation,trg_mask):
        position = torch.arange(0, image_tensor.size(1), dtype=torch.long,
                                    device=image_tensor.device)
        x = image_tensor+self.embedP(position)+self.embedX(operation)*0
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)