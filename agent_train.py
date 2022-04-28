import torch
import torchvision
from PIL import Image
import numpy as np
import time
import json
from models.config import GPT2Config, TransformerConfig
from utils import create_masks
import torch.nn.functional as F

import os
from pathlib import Path
from models.strategy import Agent
from models.get_model import get_state_model

def get_reward(real_output,state_list,state_reward_dict):
    _, samples = torch.topk(real_output, k=1, dim=-1)
    samples_np = samples.cpu().numpy()
    reward=np.ones_like(samples_np[0, :, 0])
    reward=reward.astype(np.float32)
    for index in range(samples_np.shape[1]):
        state = state_list[samples_np[0, index, 0]]
        score = state_reward_dict[state]
        reward[index]=score

    return reward

state_dict={'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
state_reward_dict={'击杀小兵或野怪或推掉塔': 2, '击杀敌方英雄': 5, '被击塔攻击': -0.5, '被击杀': -2,'无状态':0.01, '死亡': 0.01, '其它': -0.003,'普通': 0.01}
state_list=list(state_dict.keys())

train_data_path='../训练数据样本/未用/'
Path(train_data_path).mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(train_data_path):
    if len(dirs)>0:
        break
# word2numpath="./json/word2num_dict.json"
# num2word_dictpath="./json/num2word_dict.json"
# if os.path.isfile(word2numpath) and os.path.isfile(num2word_dictpath):
#     word2num_dict, num2word_dict = get_word_num_dict(word2numpath, num2word_dictpath)
# with open(word2numpath, encoding='utf8') as f:
#     word2num=json.load(f)
# config = TransformerConfig()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model_weights = "./weights/model_weights_StateJudgmentPreEn.pth"
model_get_state = get_state_model(device,model_weights,requires_grad=False)
N = 15000 # 运行N次后开始学习
epochs = 100
agent = Agent(action_num=7, input_dim=6, model_name='./weights/model_weights_agent.pth',batch_size=128,lr=0.0003, epoch=3)
block_size=600
time_start=time.time()
for epoch in range(epochs):
    #random.shuffle(dirs)
    for dir_item in os.listdir(train_data_path):
        pre_data = os.path.join(train_data_path,dir_item,'image_operationpre_data2.npz')
        if not os.path.isfile(pre_data):
            continue
        npz_file = np.load(pre_data, allow_pickle=True)
        image_tensor_np, operation_squence_np = npz_file["images_feature_tensor"], npz_file["operation_squence"]
        if image_tensor_np.shape[0]<block_size:
            continue
        operation_squence_np=np.insert(operation_squence_np,0,128)
        for cursor in range(0,operation_squence_np.shape[0],block_size):
            if (operation_squence_np.shape[0]-cursor)>600:
                operation_squence = np.array([operation_squence_np[cursor:cursor + block_size]])
                output_squence = np.array([operation_squence_np[cursor + 1:cursor + 1 + block_size]])
                image_squence = np.array([image_tensor_np[cursor:cursor + block_size, :]])
            else:
                operation_squence = np.array([operation_squence_np[-block_size-1:-1]])
                output_squence = np.array([operation_squence_np[-block_size:]])
                image_squence = np.array([image_tensor_np[-block_size:, :]])

            operation_squenceA = np.ones_like(operation_squence)

            operation_squenceA_torch = torch.from_numpy(operation_squenceA).cuda(device)
            image_squence_torch = torch.from_numpy(image_squence).cuda(device)
            output_squence_torch = torch.from_numpy(output_squence).cuda(device)
            operation_squence_torch = torch.from_numpy(operation_squence).cuda(device)
            if image_squence_torch.shape[0]!=operation_squence_torch.shape[0]:
                continue

            _, trg_mask = create_masks(operation_squence_torch, operation_squence_torch, device)
            data_dict={}
            data_dict['operation_squence'] = operation_squence
            data_dict['image_tensor'] = image_squence
            data_dict['trg_mask'] = trg_mask
            
            real_output, _ = model_get_state(image_squence_torch, operation_squenceA_torch, trg_mask)
            reward = get_reward(real_output,state_list,state_reward_dict)

            action, action_possibility, evaluate = agent.choose_action_batch(data_dict, device, output_squence_torch, True)
            loss_total = agent.supervise_strengthen_learn(device,data_dict,reward,action,action_possibility,evaluate)
          
            batch_num = cursor//block_size
            if  batch_num % 1 == 0:
                #print(loss)
                time_end = time.time()
                time_diff = time_end - time_start
                print("总时间:{} 第{}轮 第{}张 文件:{} 损失:{}".format(time_diff, epoch, batch_num, dir_item,loss_total))

    agent.save_model(epoch)
    #torch.save(model.state_dict(), 'weights/model_weights_2021-05-7D')
    #torch.save(model.state_dict(), 'weights/model_weights_2021-05-7D{}'.format(str(j)))
