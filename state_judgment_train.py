import torch
import torchvision
import json
from PIL import Image
from models.my_resnet import myResnet
import numpy as np
import torch.nn as nn
from models.sub_layers import Norm, FnLayer
import math
import torch.nn.functional as F
from models.strategy import  Transformer

from dataset.image_dataloader import get_image_feature_tensor
from utils import *

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101=torchvision.models.resnet101(pretrained=True).eval()
resnet101=myResnet(resnet101).cuda(device).requires_grad_(False)

# resnet18=torchvision.models.resnet18(pretrained=True).eval()
# resnet18=myResnet(resnet18).cuda(device).requires_grad_(False)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class StateJudgment(nn.Module):
    
    def __init__(self, class_num, out_channels, in_channels=2048,input_scale=36):
            super().__init__()
            self.out_channels=out_channels
            self.in_channels=in_channels
            self.input_scale = input_scale
            self.in_layer = FnLayer(in_channels, out_channels)
            self.hidden_layer = FnLayer(out_channels, out_channels)
            self.out_layer = FnLayer(out_channels*input_scale, class_num)
        
    def forward(self, x):
            x = x.reshape((x.shape[0], self.input_scale,self.out_channels))
            middle=gelu(self.in_layer (x))
            middle=self.hidden_layer(middle)
            middle=middle.reshape((middle.shape[0],self.out_channels*self.input_scale))
            result=self.out_layer (middle)
            return result

#model_get_state=StateJudgment(6,1024,2048).cuda(device)
model_get_state=Transformer(6,768,2,12,0.0,6*6*2048).cuda(device)
# model_get_state.load_state_dict(torch.load('weights/model_weights_StateJudgmentPreEn.pth'))
optimizer = torch.optim.Adam(model_get_state.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)

pathjson = "../训练数据样本/未用/1608391113/_操作数据.json"
all_data_dict={}
state_dict={'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
state_list=list(state_dict.keys())

with open(pathjson, encoding='ansi') as f:
    while True:
        df = f.readline()
        df = df.replace('\'', '\"')
        if df == "":
            break
        unit = json.loads(df)
        for key in unit:
            all_data_dict[key]=unit[key]
            
state = np.ones((1, ), dtype='int64')
for i in range(100):
    shuffle_dict=random_dic(all_data_dict)
    for key in shuffle_dict:
        state_dict_num=state_dict[all_data_dict[key]]
        state[0]=state_dict_num
        output=torch.from_numpy(state).cuda(device)
        image_path = '../判断数据样本/' + key + '.jpg'

        image_feature_tensor = get_image_feature_tensor(resnet101,image_path,device)

        operation_squence=np.ones((1,1))
        operation_tensor = torch.from_numpy(operation_squence.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(operation_tensor.unsqueeze(0), operation_tensor.unsqueeze(0), device)
        real_output,_=model_get_state(image_feature_tensor.unsqueeze(0), operation_tensor.unsqueeze(0),trg_mask)
        _, samples = torch.topk(real_output, k=1, dim=-1)
        samples_np = samples.cpu().numpy()
        optimizer.zero_grad()
        real_output = real_output.view(-1, real_output.size(-1))
        loss = F.cross_entropy(real_output, output.contiguous().view(-1), ignore_index=-1)
        print('epoch:', i, 'real_output', state_list[samples_np[0, 0, 0, 0]], 'output', all_data_dict[key],loss)
        loss.backward()
        optimizer.step()
    # torch.save(model_get_state.state_dict(), 'weights/model_weights_StateJudgmentL')
    torch.save(model_get_state.state_dict(), 'weights/model_weights_StateJudgmentL{}'.format(str(i)))
