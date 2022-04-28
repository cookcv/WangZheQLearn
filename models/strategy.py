import os
import numpy as np
import torch
from torch.distributions.categorical import Categorical
# distributions概率distributed和采样函数
import torch
import torch.nn as nn
import os.path
from models.config import TransformerConfig
import torch.nn.functional as F
from utils import create_masks
from models.state_model import StateModel
# from print_utils import print_sample_data
from dataset.ppo_dataloader import PPO_Dataset

def get_model(opt, trg_vocab, model_weights):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1
    model = StateModel(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    if opt.load_weights is not None and os.path.isfile(opt.load_weights + '/' + model_weights):
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/' + model_weights))
    else:
        count = 0
        for p in model.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                a = 0
            length = len(p.shape)
            point = 1
            for j in range(length):
                point = p.shape[j] * point
            count += point
        print('使用参数:{}百万'.format(count / 1000000))
    return model


class Agent:
    
    def __init__(self, action_num, input_dim, model_name, 优势估计参数G=0.9999, lr=0.0003, 泛化优势估计参数L=0.985,
                    shear=0.2, batch_size=64, epoch=10,entropy=0.01):
        self.优势估计参数G = 优势估计参数G
        self.shear = shear
        self.epoch = epoch
        self.entropy=entropy
        self.泛化优势估计参数L = 泛化优势估计参数L
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        config = TransformerConfig()
        model = get_model(config, 130, model_name)
        model = model.cuda(device)
        self.action = model
        #torch.save(self.action.state_dict(), 'weights/模型_actionppo阶段停bZ1')
        self.optimizer = torch.optim.Adam(self.action.parameters(), lr=2e-5, betas=(0.9, 0.95), eps=1e-9)
        self.dataset = PPO_Dataset(batch_size)
        self.file_name_list=[]
        
    def record_data(self, state_dict, action, action_possibility, evaluate, reward, over,count):
        self.dataset.record_data(state_dict, action, action_possibility, evaluate, reward, over,count)
        
    def save2disk(self, file_name):
        self.dataset.save2disk(file_name)
        self.file_name_list.append(file_name)
        
    def read_disk(self, file_name):
        self.dataset.read_disk(file_name)
        
    def save_model(self,epoch):
        print('... save_model ...')
        # torch.save(self.action.state_dict(), 'weights/模型_策略梯度_丙N')
        torch.save(self.action.state_dict(), 'weights/模型_策略梯度_丙N{}'.format(epoch))
        #torch.save(self.评论.state_dict(), 'weights/模型_评论')
        #torch.save(self.评论.state_dict(), 'weights/模型_评论2')
    
    def load_model(self):
        print('... load_model ...')
        # self.action.载入权重()
        #self.evaluate.载入权重()
        
    def choose_action(self, data_dict,device,input_action,manual=False):
        # distributed,q_ = self.action(state_dict)
        # r_, value = self.评论(state_dict)
        self.action.requires_grad_(False)
        operation_list=torch.from_numpy(data_dict['operation_squence'].astype(np.int64)).cuda(device)
        image_tensor=torch.from_numpy(data_dict['image_tensor']).cuda(device)
        trg_mask=data_dict['trg_mask']
        distributed, value = self.action(image_tensor,operation_list,trg_mask)
        value = value[:, - 1, :]
        distributed = F.softmax(distributed, dim=-1)
        distributed = distributed[:, - 1, :]
        distributed = Categorical(distributed)
        if manual:
            action = input_action
        else:
            action = distributed.sample()
        action_possibility = torch.squeeze(distributed.log_prob(action)).item()
        action = torch.squeeze(action).item()
        return action, action_possibility, value
        
    def choose_action_batch(self, data_dict,device,output_score_torch,manual=False):
        # distributed,q_ = self.action(state_dict)
        # r_, value = self.评论(state_dict)
        self.action.requires_grad_(False)
        operation_list=torch.from_numpy(data_dict['operation_squence'].astype(np.int64)).cuda(device)
        image_tensor=torch.from_numpy(data_dict['image_tensor']).cuda(device)
        trg_mask=data_dict['trg_mask']
        distributed, value = self.action(image_tensor,operation_list,trg_mask)
        distributed = F.softmax(distributed, dim=-1)
        distributed = Categorical(distributed)
        if manual:
            action = output_score_torch
        else:
            action = distributed.sample()
        action_possibility = torch.squeeze(distributed.log_prob(action))
        action = torch.squeeze(action)
        return action, action_possibility, value
        
    def learn(self,device):

        for _ in range(self.epoch):
            action_list, old_action_probability_list, evaluate_list, reward_list, over_list,image_list,action_num_group, batch_list = self.dataset.get_data()
            print('reward_list',reward_list[0:10])
            value = evaluate_list
            advantage_function = np.zeros(len(reward_list), dtype=np.float32)
            for t in range(len(reward_list) - 1):
                discount_rate = 1
                advantage_value = 0
                discount_rate = self.优势估计参数G * self.泛化优势估计参数L
                count=0
                for k in range(t, len(reward_list) - 1):
                    advantage_value += pow(discount_rate, abs(0-count)) * (reward_list[k] + self.优势估计参数G * value[k + 1] * (1 - int(over_list[k])) - value[k])
                    count=count+1
                    if (1 - int(over_list[k]))==0 or count>100:
                        break
                advantage_function[t] = advantage_value
                # https://blog.csdn.net/zhkmxx930xperia/article/details/88257891
                # GAE的形式为多个value估计的加权平均数
            advantage_function = torch.tensor(advantage_function).to(device)
            value = torch.tensor(value).to(device)
            for batch_item in batch_list:
                batch_item_end=batch_item[-1:]
                old_action_possibilitys = torch.tensor(old_action_probability_list[batch_item_end]).to(device)
                actions = torch.tensor(action_list[batch_item_end]).to(device)
                self.action.requires_grad_(True)
                operation_list = torch.from_numpy(action_num_group[batch_item].astype(np.int64)).cuda(device)
                image_tensor = torch.from_numpy(image_list[:, batch_item, :]).cuda(device).float()
                src_mask, trg_mask = create_masks(operation_list.unsqueeze(0), operation_list.unsqueeze(0), device)
                distributed, evaluate = self.action(image_tensor,operation_list,trg_mask)
                distributed=distributed[:,-1:,:]
                evaluate = evaluate[:, -1:, :]
                distributed = F.softmax(distributed, dim=-1)
                # distributed = distributed[:, - 1, :]
                # evaluate = evaluate[:, - 1, :]
                evaluate = torch.squeeze(evaluate)
                distributed = Categorical(distributed)
                entropy_loss = torch.mean(distributed.entropy())
                new_action_possibilitys = distributed.log_prob(actions)
                # 概率比 = new_action_possibilitys.exp() / old_action_possibilitys.exp()
                # # prob_ratio = (new_probs - old_probs).exp()
                # 加权概率 = advantage_function[batch_item_end] * 概率比
                # 加权_裁剪_概率 = torch.clamp(概率比, 1 - self.shear,
                #                                  1 + self.shear) * advantage_function[batch_item_end]
                # action_loss = -torch.min(加权概率, 加权_裁剪_概率).mean()
                reward_total = advantage_function[batch_item_end] + value[batch_item_end]
                action_loss = -reward_total * new_action_possibilitys
                action_loss = action_loss.mean()
                evaluate_loss = (reward_total - evaluate) ** 2
                evaluate_loss = evaluate_loss .mean()
                loss_total = action_loss + 0.5 * evaluate_loss-self.entropy*entropy_loss
                #print(loss_total)
                self.optimizer.zero_grad()
            # self.optimizer_评论.zero_grad()
                loss_total.backward()
                self.optimizer.step()
            # self.optimizer_评论.step()
            # print('loss_total',loss_total)
        self.dataset.delete_data()
        self.file_name_list=[]
        
    def supervise_strengthen_learn(self,device,data_dict,reward,action,action_possibility,evaluate):

        reward_list=reward
        value=evaluate.cpu().numpy()[0,:,0]
        advantage_function = np.zeros(reward_list.shape[0], dtype=np.float32)
        for t in range(len(reward_list) - 1):
            discount_rate = 1
            advantage_value = 0
            discount_rate = self.优势估计参数G * self.泛化优势估计参数L
            count = 0
            for k in range(t, len(reward_list) - 1):
                advantage_value += pow(discount_rate, abs(0 - count)) * (reward_list[k])
                count = count + 1
                if count > 200:
                    break
            advantage_function[t] = advantage_value
            value = torch.as_tensor(value).to(device)
        for i in range(self.epoch):
            advantage_function = torch.as_tensor(advantage_function).to(device) #advantage_function.clone().detach().to(device)
            # old_action_possibilitys = torch.tensor(action_possibility).to(device)
            actions = torch.as_tensor(action).to(device)
            self.action.requires_grad_(True)
            operation_list = torch.from_numpy(data_dict['operation_squence'].astype(np.int64)).cuda(device)
            image_tensor = torch.from_numpy(data_dict['image_tensor']).cuda(device).float()
            trg_mask = data_dict['trg_mask']
            distributed, evaluate = self.action(image_tensor, operation_list, trg_mask)
            distributed = F.softmax(distributed, dim=-1)
            # distributed = distributed[:, - 1, :]
            # evaluate = evaluate[:, - 1, :]
            evaluate = torch.squeeze(evaluate)
            distributed = Categorical(distributed)
            #entropy_loss = torch.mean(distributed.entropy())
            new_action_possibilitys = distributed.log_prob(actions)
            # old_action_possibilitys=old_action_possibilitys.exp()
            # 概率比 = new_action_possibilitys / old_action_possibilitys
            # # prob_ratio = (new_probs - old_probs).exp()
            # 加权概率 = advantage_function * 概率比
            # 加权_裁剪_概率 = torch.clamp(概率比, 1 - self.shear,
            #                    1 + self.shear) * advantage_function
            # action_loss = -torch.min(加权概率, 加权_裁剪_概率).mean()
            #概率比2 = new_action_possibilitys.mean() / old_action_possibilitys.mean()
            reward_total = advantage_function#+ value
            action_loss = -reward_total * new_action_possibilitys
            action_loss = action_loss.mean()
            #evaluate_loss = (reward_total - evaluate) ** 2
            #evaluate_loss = evaluate_loss.mean()
            # print(reward_total[10:20],new_action_possibilitys[:,10:20].exp())
            loss_total = action_loss# + 0.5 * evaluate_loss - self.entropy * entropy_loss
            # print(loss_total)
            self.optimizer.zero_grad()
            # self.optimizer_评论.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            return loss_total
        # self.optimizer_评论.step()
        
    def supervise_strengthen_learnA(self,device,data_dict,reward,action,action_possibility,evaluate,over_list):

        reward_list=reward
        value=evaluate.cpu().numpy()[0,:,0]
        advantage_function = np.zeros(reward_list.shape[0], dtype=np.float32)
        for t in range(len(reward_list) - 1):
            discount_rate = 1
            advantage_value = 0
            discount_rate = self.优势估计参数G * self.泛化优势估计参数L
            count = 0
            for k in range(t, len(reward_list) - 1):
                advantage_value += pow(discount_rate, abs(0 - count)) * (reward_list[k]*(1-over_list[0,k]*0))
                count = count + 1
                if  count > 200 or over_list[0,k]==2111111:
                    break
            advantage_function[t] = advantage_value
            value = torch.tensor(value).to(device)
        for i in range(3):
            advantage_function = torch.tensor(advantage_function).to(device)
            old_action_possibilitys = torch.tensor(action_possibility).to(device)
            actions = torch.tensor(action).to(device)
            self.action.requires_grad_(True)
            operation_list = torch.from_numpy(data_dict['operation_squence'].astype(np.int64)).cuda(device)
            image_tensor = torch.from_numpy(data_dict['image_tensor']).cuda(device).float()
            trg_mask = data_dict['trg_mask']
            distributed, evaluate = self.action(image_tensor, operation_list, trg_mask)
            distributed = F.softmax(distributed, dim=-1)
            # distributed = distributed[:, - 1, :]
            # evaluate = evaluate[:, - 1, :]
            evaluate = torch.squeeze(evaluate)
            distributed = Categorical(distributed)
            #entropy_loss = torch.mean(distributed.entropy())
            new_action_possibilitys = distributed.log_prob(actions)
            # old_action_possibilitys=old_action_possibilitys.exp()
            # 概率比 = new_action_possibilitys / old_action_possibilitys
            # # prob_ratio = (new_probs - old_probs).exp()
            # 加权概率 = advantage_function * 概率比
            # 加权_裁剪_概率 = torch.clamp(概率比, 1 - self.shear,
            #                    1 + self.shear) * advantage_function
            # action_loss = -torch.min(加权概率, 加权_裁剪_概率).mean()
            #概率比2 = new_action_possibilitys.mean() / old_action_possibilitys.mean()
            reward_total = advantage_function#+ value
            action_loss = -reward_total * new_action_possibilitys
            action_loss = action_loss.mean()
            #evaluate_loss = (reward_total - evaluate) ** 2
            #evaluate_loss = evaluate_loss.mean()
            print(reward_total[10:20],new_action_possibilitys[:,10:20].exp())
            loss_total = action_loss# + 0.5 * evaluate_loss - self.entropy * entropy_loss
            # print(loss_total)
            self.optimizer.zero_grad()
            # self.optimizer_评论.zero_grad()
            loss_total.backward()
            self.optimizer.step()
        # self.optimizer_评论.step()
        
    def supervise_learn(self, state_dict,output,print_out,num2word_dict,operation_score_torch,device):
        distributed, value = self.action(state_dict,device)
        lin = distributed.view(-1, distributed.size(-1))
        _, samples = torch.topk(distributed, k=1, dim=-1)
        samples_np = samples.cpu().numpy()
        self.optimizer.zero_grad()
        loss = F.cross_entropy(lin, output.contiguous().view(-1), ignore_index=-1)
        if print_out:
            print(loss)
            # print_sample_data(num2word_dict, samples_np[0:1, :, :], operation_score_torch[0, :])
        loss.backward()
        self.optimizer.step()
        
    def choose_action_batch_old(self, state_dict):
        # distributed,q_ = self.action(state_dict)
        # r_, value = self.评论(state_dict)
        real_output_A, value = self.action(state_dict)
        real_output_A = F.softmax(real_output_A, dim=-1)
        real_output_A = real_output_A[:, - 1, :]
        samples = torch.multinomial(real_output_A, num_samples=1)
        samples_np = samples.cpu().numpy()
        return  samples_np[0,-1]
    #item是得到一个元素张量里面的元素值
    #优势函数表达在state_dicts下，某actiona相对于平均而言的优势
    #GAE一般优势估计
