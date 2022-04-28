import json
import numpy as np

def get_train_data(data_path):
    input_list = list
    output_list = list
    with open(data_path,encoding='utf-8') as f:
        while True:
            data_line = f.readline()
            if not data_line:
                break
            json_line = json.loads(data_line)
            json_data = json_line['data']
            data_input = json_data['input']
            data_output = json_data['output']
            item_len = len(data_input)//16
            for i in range(16):
                input_list.append(data_input[i*item_len:(i+1)*item_len])
                output_list.append(data_output[i*item_len:(i+1)*item_len])
    return input_list,output_list

def 写出词标号引索(总词表,  word2num_dictpath, num2word_dictpath):
    print("正在写出词的标号引索数据可能需要较长时间")
    标号_到_字符 = {}
    字符_到_标号 = {}
    标号_字符 = []
    # 标号_到_字符 = list(set(总表单))
    i = 0
    j = 0
    for 词表 in 总词表:
        j = j + 1
        for 字符 in 词表:
            if 字符 not in 标号_字符:
                标号_字符.append(字符)
                字符_到_标号[字符] = i
                标号_到_字符[i] = 字符
                i = i + 1
        if j % 10000 == 0:
            print(i, 标号_到_字符[i - 1],  j/len(总词表))
    #print(标号_到_字符[1], 标号_到_字符[111], len(标号_到_字符))
    with open(word2num_dictpath, 'w', encoding='utf-8') as f:
        json.dump(字符_到_标号, f, ensure_ascii=False)
    with open(num2word_dictpath, 'w', encoding='utf-8') as f:
        json.dump(标号_到_字符, f, ensure_ascii=False)

def get_word_num_dict(word2num_dictpath, num2word_dictpath):
    with open(word2num_dictpath, encoding='utf-8') as f:
        word2num_dict= json.load(f)
    with open(num2word_dictpath, encoding='utf-8') as f:
        num2word_dict = json.load(f)
    return word2num_dict, num2word_dict

def 生成训练用numpy数组(输入表单, word2num_dict, numpy_array_path):
    表_1 = []
    表_2 = []
    i = 0
    临 = ''
    for 表单 in 输入表单:
        表_3 = []
        for 字符 in 表单:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':
                    临 = 字符
                else:
                    临 = 临 + 字符
            else:
                if 临 == '':
                    if 字符.lower() in word2num_dict:
                         表_3.append(word2num_dict[字符.lower()])
                    else:
                        表_3.append(14999)
                else:
                    if 临.lower() in word2num_dict:
                        表_3.append(word2num_dict[临.lower()])
                    else:
                        表_3.append(14999)
                    临 = ''
                    if 字符.lower() in word2num_dict:
                        表_3.append(word2num_dict[字符.lower()])
                    else:
                        表_3.append(14999)
        if 临 != '':
            if 临.lower() in word2num_dict:
                表_3.append(word2num_dict[临.lower()])
            else:
                表_3.append(14999)
            临 = ''
        if len(表_3) != 667:
            # 表_1.append(np.array(表_3[0:-1]))
            # 表_2.append(np.array(表_3[1:]))
            print(表_3)
        else:
            表_1.append(np.array(表_3[0:-1]))
            表_2.append(np.array(表_3[1:]))
        if i % 1000 == 0:
            print("数据转化为numpy数组完成度百分比{}".format(i / len(输入表单) * 100))
        i = i + 1
    print("数据转化为numpy数组完成。")
    输入np = np.array(表_1)
    输出np = np.array(表_2)
    np.savez(numpy_array_path, 输出np=输出np, 输入np=输入np)

def 生成测试用numpy数组(输入表单, word2num_dict):
    表_1 = []
    for 字符 in 输入表单:
        if 字符.lower() in word2num_dict:
            表_1.append(word2num_dict[字符])
        else:
            表_1.append(14999)
    输入np = np.array(表_1)
    return (输入np)

def 生成训练用numpy数组_A(输入表单,  word2num_dict, numpy_array_path):
    表_1 = []
    表_2 = []
    i=0
    临=''
    for 表单 in 输入表单:
        表_3=[]
        for 字符 in 表单:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':
                    临 = 字符
                else:
                    临 = 临 + 字符
            else:
                if 临 == '':
                    if 字符.lower() in word2num_dict:
                        if 字符 != ' ':
                            表_3.append(word2num_dict[字符.lower()])
                    else:
                        表_3.append(14999)
                else:
                    if 临.lower() in word2num_dict:
                        if 临 != ' ':
                            表_3.append(word2num_dict[临.lower() ])
                    else:
                        表_3.append(14999)
                    临=''
                    if 字符.lower() in word2num_dict:
                        if 字符 != ' ':
                            表_3.append(word2num_dict[字符.lower() ])
                    else:
                        表_3.append(14999)
        if 临!='':
            if 临.lower() in word2num_dict:
                if 字符 != ' ':
                    表_3.append(word2num_dict[临.lower() ])
            else:
                表_3.append(14999)
            临 = ''
        if len(表_3)!=667:
            #表_1.append(np.array(表_3[0:-1]))
            #表_2.append(np.array(表_3[1:]))
            print(表_3)
        else:
            表_1.append(np.array(表_3[0:-1]))
            表_2.append(np.array(表_3[1:]))
        if i % 1000 == 0:
            print("数据转化为numpy数组完成度百分比{}".format(i/len(输入表单)*100))
        i = i + 1
    print("数据转化为numpy数组完成。")
    输入np = np.array(表_1)
    输出np = np.array(表_2)
    np.savez(numpy_array_path, 输出np=输出np, 输入np=输入np)

def 读取训练数据_A(path):
    输入表单 = []
    with open(path, encoding='utf-8') as f:
        while True:
            行 = f.readline()
            if not 行:
                break
            json_行 = json.loads(行)
            内容 = json_行['input']
            输入表单.append(内容)
    return 输入表单

def 生成测试用numpy数组_A(输入表单, word2num_dict):
    表_3 = []
    临 = ''
    for 字符 in 输入表单:
        if 字符.lower() in word2num_dict:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':
                    临 = 字符
                else:
                    临 = 临 + 字符
            else:
                if 临 == '':
                    if 字符.lower() in word2num_dict:
                        if 字符.lower() != ' ':
                            表_3.append(word2num_dict[字符.lower()])
                    else:
                        表_3.append(14999)
                else:
                    if 临.lower() in word2num_dict:
                        if 临.lower() != ' ':
                            表_3.append(word2num_dict[临.lower()])
                    else:
                        表_3.append(14999)
                    临 = ''
                    if 字符.lower() in word2num_dict:
                        if 字符.lower() != ' ':
                            表_3.append(word2num_dict[字符.lower()])
                    else:
                        表_3.append(14999)
    输入np = np.array(表_3)
    return (输入np)