import torch
import numpy as np
import pickle
from utils import create_masks

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)

class PPO_Dataset:
    
    def __init__(self, batch_size):
        #self.state_dict集 = []
        self.action_probability_list = []
        self.evaluate_list = []
        self.action_list = []
        self.reward_list = []
        self.over_list = []
        self.batch_size = batch_size
        self.all_data_dict={}
        self.image_info=np.ones([1,1000, 6*6*2048], dtype='float')
        self.operation_info = np.ones((0,))
        
    def get_data(self):
        state_dict_len = len(self.reward_list)
        batch_start_index = np.arange(0, state_dict_len-100, self.batch_size)
        index_list = np.arange(state_dict_len, dtype=np.int64)
        batch_list = [index_list[i:i + self.batch_size] for i in batch_start_index]
        return  np.array(self.action_list),\
                np.array(self.action_probability_list), \
                self.evaluate_list, \
                np.array(self.reward_list),\
                np.array(self.over_list), \
                self.image_info, \
                self.operation_info ,\
                batch_list
        
    def record_data(self, data_dict, action, action_possibility, evaluate, reward, over,count):
        #self.state_dict集.append(state_dict)
        self.action_list.append(action)
        self.action_probability_list.append(action_possibility)
        self.evaluate_list.append(evaluate)
        self.reward_list.append(reward)
        self.over_list.append(over)
        self.image_info[:,count, :]=data_dict['image_tensor']
        self.operation_info=np.append(self.operation_info, data_dict['operation_squence'])
        
    def delete_data(self):
        self.image_info = []
        self.action_probability_list = []
        self.action_list = []
        self.reward_list = []
        self.over_list = []
        self.evaluate_list = []
        self.all_data_dict={}
        # del self.state_dict集,self.action_probability_list,self.evaluate_list,self.action_list,self.reward_list,self.over_list,self.all_data_dict
        # gc.collect()
        
    def save2disk(self,file_name):
        self.all_data_dict['image_info']=self.image_info[:,0:len(self.action_list),:]
        self.all_data_dict['action_probability_list'] = self.action_probability_list
        self.all_data_dict['action_list'] = self.action_list
        self.all_data_dict['reward_list'] = self.reward_list
        self.all_data_dict['over_list'] = self.over_list
        self.all_data_dict['evaluate_list'] = self.evaluate_list
        self.all_data_dict['operation_info'] =self.operation_info
        save_obj(self.all_data_dict,file_name)
        self.all_data_dict={}
        #self.image_info = []
        self.action_probability_list = []
        self.action_list = []
        self.reward_list = []
        self.over_list = []
        self.evaluate_list = []
        #self.operation_info=[]
        #del self.image_info,self.action_probability_list,self.evaluate_list,self.action_list,self.reward_list,self.over_list,self.all_data_dict
        #gc.collect()
        
    def read_disk(self,file_name):
        self.all_data_dict = load_obj(file_name)
        self.image_info=self.all_data_dict['image_info']
        self.action_probability_list=self.all_data_dict['action_probability_list']
        self.action_list=self.all_data_dict['action_list']
        self.reward_list= self.all_data_dict['reward_list']
        self.over_list= self.all_data_dict['over_list']
        self.evaluate_list=self.all_data_dict['evaluate_list']
        self.operation_info =self.all_data_dict ['operation_info']
        self.all_data_dict={}

    def state_dict_param_process(state_group,device):
        max_length=0
        state_combination={}
    # operation_list = np.ones((1,))
        for state_dictA in state_group:
            if state_dictA['image_tensor'].shape[1]>max_length:
                max_length=state_dictA['image_tensor'].shape[1]
        for state_dict in state_group:
            state_dictA = state_dict.copy()
            if state_dictA['image_tensor'].shape[1] == max_length:
                unit=state_dictA
                operation_list = np.ones((max_length,))
                mask_sequence = torch.from_numpy(operation_list.astype(np.int64)).cuda(device).unsqueeze(0)
                unit['mask_sequence']=mask_sequence
            else:
                effective_length=state_dictA['image_tensor'].shape[1]
                length_diff=max_length-effective_length
                image_shape=state_dictA['image_tensor'].shape
                image_tensor_concate = torch.zeros(image_shape[0],length_diff,image_shape[2],image_shape[3]).cuda(device).float()
                image_tensor_concate = image_tensor_concate.cpu().numpy()
                state_dictA['image_tensor']=np.append(state_dictA['image_tensor'],image_tensor_concate, axis=1)
                #state_dictA['image_tensor'] = torch.cat((state_dictA['image_tensor'], image_tensor_concate), 1)
                image_shape = state_dictA['angle_tensor_squence'].shape
                angle_tensor_concate=torch.zeros(image_shape[0],length_diff,image_shape[2]).cuda(device).float()
                state_dictA['angle_tensor_squence'] = torch.cat((state_dictA['angle_tensor_squence'], angle_tensor_concate), 1)
                image_shape = state_dictA['position_tensor_squence'].shape
                position_tensor_concate=torch.zeros(image_shape[0],length_diff,image_shape[2]).cuda(device).float()
                state_dictA['position_tensor_squence'] = torch.cat((state_dictA['position_tensor_squence'], position_tensor_concate), 1)
                image_shape = state_dictA['speed_tensor_squence'].shape
                speed_tensor_concate=torch.zeros(image_shape[0],length_diff,image_shape[2]).cuda(device).float()
                state_dictA['speed_tensor_squence'] = torch.cat((state_dictA['speed_tensor_squence'], speed_tensor_concate), 1)
                operation_list = np.ones((effective_length,))
                mask_sequence = torch.from_numpy(operation_list.astype(np.int64)).cuda(device).unsqueeze(0)
                state_dictA['mask_sequence']=mask_sequence
                operation_list = np.ones((length_diff,))*-1
                mask_sequence = torch.from_numpy(operation_list.astype(np.int64)).cuda(device).unsqueeze(0)
                state_dictA['mask_sequence'] = torch.cat((state_dictA['mask_sequence'], mask_sequence), 1)
                unit=state_dictA
            if state_combination=={}:
                state_combination=unit
            else:
                state_combination['mask_sequence'] = torch.cat((state_combination['mask_sequence'], unit['mask_sequence']), 0)
                state_combination['speed_tensor_squence'] = torch.cat((state_combination['speed_tensor_squence'], unit['speed_tensor_squence'],), 0)
                state_combination['position_tensor_squence'] = torch.cat((state_combination['position_tensor_squence'], unit['position_tensor_squence']), 0)
                state_combination['angle_tensor_squence'] = torch.cat((state_combination['angle_tensor_squence'], unit['angle_tensor_squence']), 0)
                #state_combination['image_tensor'] = torch.cat((state_combination['image_tensor'], unit['image_tensor']), 0)
                state_combination['image_tensor'] = np.append(state_combination['image_tensor'], unit['image_tensor'], axis=0)
        src_mask, trg_mask = create_masks(state_combination['mask_sequence'], state_combination['mask_sequence'], device)
        state_combination['trg_mask']=trg_mask
        return state_combination
