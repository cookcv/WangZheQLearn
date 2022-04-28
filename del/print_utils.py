
from torch.autograd import Variable
import torch
import numpy as np

def print_sample_data(num2word_dict,数据, 输出_score):
    临 = 数据[0]
    欲打印=[num2word_dict[str(临[i,0])] for i in range(0,临.shape[0])]
    临 = 输出_score.cpu().numpy()
    欲打印2 = [num2word_dict[str(临[i])] for i in range(0,临.shape[0])]
    print("samples输出",欲打印)
    print("output", 欲打印2)
    # for i in range(16):
    #     print(num2word_dict[str(临[i, 0])])

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda(device)
    return np_mask

def print_test_data(num2word_dict,数据, 输人_score,标签):
    临 = 数据[0]
    欲打印=[num2word_dict[str(临[i])] for i in range(临.size)]
    打印=""
    for i in range(len(欲打印)):
        打印=打印+欲打印[i]
    临 = 输人_score.cpu().numpy()[0]
    欲打印2 = [num2word_dict[str(临[i])]for i in range(输人_score.size(1))]
    # 欲打印2=str(欲打印2)
    # print("输入：", 欲打印2)
    if 标签==打印:
        return True
    else:
        print(打印)
        return False
    print("输出：",打印)
    # for i in range(16):
    #     print(num2word_dict[str(临[i, 0])])

def print_test_data_A(num2word_dict,数据, 输人_score):
    if 数据.shape[0]!=0:
        临 = 数据[0]
        欲打印=[num2word_dict[str(临[i])] for i in range(临.size)]
        打印=""
        for i in range(len(欲打印)):
            打印=打印+欲打印[i]
        临 = 输人_score.cpu().numpy()[0]
        欲打印2 = [num2word_dict[str(临[i])]for i in range(输人_score.size(1))]
        欲打印2=str(欲打印2)
        #print("输入：", 欲打印2)
        print("输出：",打印)