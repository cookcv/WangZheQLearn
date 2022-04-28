import torch.nn as nn
from models.sub_layers import FnLayer
from models.image_feature_decoder import ImageFeatureDecoder

class StateModel(nn.Module):

    def __init__(self,  trg_vocab, d_model, N, heads, dropout,tensor_size=6*6*2048):
        super().__init__()
        self.fn_layer= FnLayer(tensor_size,d_model)
        self.decoder = ImageFeatureDecoder(trg_vocab, d_model, N, heads, dropout)
        self.outX = FnLayer(d_model, trg_vocab)
        self.evaluate = FnLayer(d_model, 1)
        
    def forward(self, image_tensor ,operation, trg_mask):
        image_tensor=self.fn_layer(image_tensor)
        
        d_output = self.decoder(image_tensor, operation, trg_mask)
        output = self.outX(d_output)
        evaluate = self.evaluate(d_output)
        return output,evaluate