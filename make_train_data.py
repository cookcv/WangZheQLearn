import os

from utils import create_masks
import json
from utils import *
from pynput.keyboard import Key
import time, threading
from models.strategy import Agent
from pathlib import Path
from models.get_model import get_feature_model
from equipment.equipment import Equipment
from PIL import Image
from dataset.json_dataloader import get_data_dict
from dataset.image_dataloader import get_image_from_phone,get_image_feature_tensor
from equipment.equipment_listen import direction_process
import torch

_DEVICE_ID = '65784b6'
window_name = "BRQ-ANN00" #"BRQ-ANN00" #"subWin"

train_data_save_path='../训练数据样本/未用1'
Path(train_data_save_path).mkdir(parents=True, exist_ok=True)
image_path = train_data_save_path+'/{}/'.format(str(int(time.time())) )
os.mkdir(image_path)
log_file=open(image_path+'_operation数据.json','w+')

num2word_dictpath="./json/num2word_dict.json"
operation_inquire_path="./json/名称_操作.json"
operation_inquire_dict = get_data_dict(operation_inquire_path)
num2word_dict = get_data_dict(num2word_dictpath)

equipment = Equipment(_DEVICE_ID,operation_inquire_dict)

def save_image_operation(file,image:Image,image_index,operation:dict):
    image_file = image_path + '{}.jpg'.format(image_index)
    image.save(image_file)
    json.dump(operation, file, ensure_ascii=False)
    file.write('\n')

lock=threading.Lock()
th = threading.Thread(target=equipment.start_listen,)
th.start() #启动线程

manual_operation_list=[]
operation_dict={"图片号":"0","移动操作":"无移动","动作操作":"无动作"}
agent = Agent(action_num=7, input_dim=6,
    model_name = 'weights_82',
    batch_size=128,lr=0.0003, epoch=3)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101 = get_feature_model(device,requires_grad=False)
while True:
    if not equipment.ai_open :
        continue

    images_feature_tensor = torch.Tensor(0)
    operation_tensor = torch.Tensor(0)
    operation_squence = np.ones((1, ))
    operation_squence[0] = 128
    count = 0

    for i in range(10000):
        if equipment.ai_open==False:
            break
        try:
            image_pil = get_image_from_phone(window_name)
        except:
            equipment.ai_open = False
            print('get_image_from_phone失败')
            break
        time_start=time.time()
        img = np.array(image_pil)
        image_feature_tensor = get_image_feature_tensor(resnet101,img,device)
        if images_feature_tensor.shape[0] == 0:
            images_feature_tensor = image_feature_tensor
        elif images_feature_tensor.shape[0] < 300:
            images_feature_tensor = torch.cat((images_feature_tensor, image_feature_tensor), 0)
            operation_squence = np.append(operation_squence, action)
        else:
            images_feature_tensor = images_feature_tensor[1:300, :]
            operation_squence = operation_squence[1:300]
            operation_squence = np.append(operation_squence, action)
            images_feature_tensor = torch.cat((images_feature_tensor, image_feature_tensor), 0)

        operation_tensor = torch.from_numpy(operation_squence.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(operation_tensor.unsqueeze(0), operation_tensor.unsqueeze(0), device)
        state = get_state_info(images_feature_tensor.cpu().numpy(), operation_squence, trg_mask)
        action, _, _ = agent.choose_action(state,device,1,False)
        move_attack_instruction = num2word_dict[str(action)]
        move_instruction,attack_instruction = move_attack_instruction.split('_')
        # LI = operation_tensor.contiguous().view(-1)
        # LA=real_output_A.view(-1, real_output_A.size(-1))
        if count % 50 == 0 and count!=0:
            equipment.buy_add_skill()
            print(equipment.present_move_instruction,'周期')

        if count % 1 == 0:
            time_end = time.time()
            operation_dict['图片号']=str(i)
            direction_result = direction_process(equipment.press_w,equipment.press_s,equipment.press_a,equipment.press_d,equipment.press_q)
            if direction_result!='' or manual_operation_list or equipment.attack_state==True:
                auto = 1
                if not direction_result:
                    operation_dict['移动操作'] = move_instruction
                else:
                    operation_dict['移动操作'] = direction_result
                if not manual_operation_list:
                    lock.acquire()
                    operation_dict['动作操作'] = manual_operation_list.pop(0)
                    lock.release()
                elif equipment.attack_state==True:
                    operation_dict['动作操作'] = '攻击'
                else:
                    operation_dict['动作操作'] = '无动作'
            else:
                auto = 0
                operation_dict['移动操作'] = move_instruction
                operation_dict['动作操作'] = attack_instruction
            
            operation_dict['自动'] = auto
            save_image_operation(log_file,image_pil,i,operation_dict)
            new_move_instruction = operation_dict['移动操作']
            new_attack_instruction = operation_dict['动作操作']
            equipment.send_move_instruction(new_move_instruction)
            equipment.send_atack_instruction(new_attack_instruction)

            time1=0.22-(time.time()-time_start)
            if time1>0:
                time.sleep(time1)
            time_consuming = time_end - time_start
            count=count+1

    log_file.close()
    time.sleep(1)
    print('AI_open',equipment.ai_open)
