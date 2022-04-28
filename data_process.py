import torch
import torchvision
import numpy as np
import os
import json
from PIL import Image
from pathlib import Path
from models.my_resnet import myResnet
from tqdm import tqdm
from dataset.image_dataloader import get_image_feature_tensor
from models.get_model import get_feature_model
from dataset.json_dataloader import get_data_dict

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
## 加载图像识别模型，将模型结果池化成2048个6x6大小的特征图，并将该结果代替原图进行保存。
resnet101 = get_feature_model(device,requires_grad=False)
# 操作对应的序列化数值
word2numpath = "./json/word2num_dict.json"
word2num = get_data_dict(word2numpath)

operation_record='../训练数据样本/未用'
for dir_item in os.listdir(operation_record):
    ## 获取每场game的操作数据
    pathjson = os.path.join(operation_record,dir_item,'_操作数据.json')
    ## 每场game的数据单独存放到一个文件中
    numpy_array_path = os.path.join(operation_record,dir_item,'image_operationpre_data2.npz')
    if os.path.isfile(numpy_array_path):
        continue
    ## 存储信息有，图像特征图、操作数值
    images_feature_tensor = torch.Tensor(0)
    operation_squence = np.ones((1, 1))

    print('正在处理{}'.format(dir_item))
    data_lines=[]
    with open(pathjson, encoding='ansi') as f:
        data_lines = f.readlines()
    ## 遍历而标注数据，将结果进行转化储存。
    for data_line in tqdm(data_lines,ncols=-1):
        data_line = data_line.replace('\'', '\"')
        df = json.loads(data_line)
        image_path = os.path.join(operation_record,dir_item,df["图片号"]+".jpg")
        input_img_pil = Image.open(image_path)
        input_img_array = np.array(input_img_pil)
        image_feature_tensor = get_image_feature_tensor(resnet101,input_img_array,device)

        move_action_operation = "{}_{}".format(df["移动操作"],df["动作操作"])
        
        if images_feature_tensor.shape[0] == 0:
            images_feature_tensor = image_feature_tensor
            operation_squence[0, 0] = word2num[move_action_operation]
        else:
            images_feature_tensor = torch.cat((images_feature_tensor, image_feature_tensor), 0) ## 图像数据集
            operation_squence = np.append(operation_squence, word2num[move_action_operation]) ## 操作数据集
           
    images_feature_tensor = images_feature_tensor.cpu().numpy()
    operation_squence=operation_squence.astype(np.int64)
    np.savez(numpy_array_path, images_feature_tensor=images_feature_tensor, operation_squence=operation_squence)
