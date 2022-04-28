import os
# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# print(device)
import json
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import shutil
from pynput.keyboard import Key
import threading
from pathlib import Path
from equipment.equipment_listen import LabelRectifyListen

state='暂停'
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        'C:/Windows/Fonts/Arial', textSize, encoding="utf-8")
    #"D:/python/辅助/锐字真言体.ttf"
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

listen = LabelRectifyListen()
th = threading.Thread(target=listen.start_listen,)
th.start()

#筛选事件特征图片
#1、进入目录打开引索 方法抄
pathjson='../判断数据样本test/_判断数据.json'
new_path='../判断数据样本/'
Path(new_path).mkdir(parents=True, exist_ok=True)
new_path = os.path.join(new_path,'判断新.json')
all_data_dict={}
with open(pathjson, encoding='ansi') as f:
    while True:
        df = f.readline()
        df = df.replace('\'', '\"')
        if df == "":
            break
        unit = json.loads(df)
        for key in unit:
            all_data_dict[key]=unit[key]
#print(all_data_dict)
for key in all_data_dict:
    log_file = open(new_path, 'a+')
   # print(key + ':' + all_data_dict[key])
    image_path = '../判断数据样本test/' + key + '.jpg'
    new_image_path = '../判断数据样本/'+ key + '.jpg'
    # screenshot = cv2.imread(image_path)
    screenshot = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    screenshot = cv2ImgAddText(screenshot, all_data_dict[key], 0, 0, (000, 222, 111), 25)
    cv2.imshow('AAA', screenshot)
    cv2.waitKey()
    while state == '暂停':
        time.sleep(0.02)
    new_output={}
    calibration_output=all_data_dict[key]
    if state=='过':
        calibration_output = all_data_dict[key]
    elif state=='普通':
        calibration_output = '普通'
    elif state == '死亡':
        calibration_output = '死亡'
    elif state == '被击杀':
        calibration_output = '被击杀'
    elif state == '击杀小兵或野怪或推掉塔':
        calibration_output = '击杀小兵或野怪或推掉塔'
    elif state == '击杀敌方英雄':
        calibration_output = '击杀敌方英雄'
    elif state == '被击塔攻击':
        calibration_output = '被击塔攻击'
    elif state == '弃' and key!='162098566208':
        state = '暂停'
        continue
    else:
        print(1)
    print(key, calibration_output)
    new_output[key]=calibration_output
    json.dump(new_output, log_file, ensure_ascii=False)
    log_file.write('\n')
    shutil.copy(image_path, new_image_path)
    state = '暂停'
    log_file.close()
