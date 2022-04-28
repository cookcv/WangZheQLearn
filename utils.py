

import numpy as np
from random import shuffle
import torch

from torch.autograd import Variable

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    variable = Variable
    np_mask = variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda(device)
    return np_mask

def create_masks(src, trg, device):
    
    src_mask = (src != -1).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != -1).unsqueeze(-2)
        trg_mask.cuda(device)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def get_state_info(image_tensor,operation_squence,trg_mask):
    state={}
    state['image_tensor'] = image_tensor[np.newaxis, :]
    state['operation_squence'] = operation_squence
    state['trg_mask'] = trg_mask
    return state

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
