import torch
import win32gui, win32ui, win32con
from PIL import Image
from PIL import ImageQt
import numpy as np

def get_image_feature_tensor(model,input_img_array,device):

    input_img_tensor = torch.from_numpy(input_img_array).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
    _, out = model(input_img_tensor)
    image_tensor = out.reshape(1, 6 * 6 * 2048)

    return image_tensor

def get_npimage_from_screen(screen,window):
    img = screen.grabWindow(window)
    image = ImageQt.fromqimage(img)
    imgA = image.resize((960, 480))
    #imgA = get_image_from_phone(窗口名称)
    image_np=np.asarray(imgA)

    return image_np

def get_image_from_phone(windows_name):
    # 获取后台窗口的句柄，注意后台窗口不能最小化
    hWnd = win32gui.FindWindow("SDL_app","vivo X20A")
    # hWnd = win32gui.FindWindow(0,windows_name)  # 窗口的类名可以用Visual Studio的SPY++工具获取
    # 获取句柄窗口的大小信息
    left, top, right, bot = win32gui.GetWindowRect(hWnd)
    width = right - left
    height = bot - top
    # 返回句柄窗口的equipment环境，覆盖整个窗口，包括非客户区，标题栏，菜单，边框
    hWndDC = win32gui.GetWindowDC(hWnd)
    # 创建equipment描述表
    mfcDC = win32ui.CreateDCFromHandle(hWndDC)
    # 创建内存equipment描述表
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建位图对象准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 为bitmap开辟存储空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    # 将screenshot保存到saveBitMap中
    saveDC.SelectObject(saveBitMap)
    # 保存bitmap到内存equipment描述表
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    ###生成图像
    im_PIL = Image.frombuffer('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr,'raw','BGRX')
    #im_PIL= Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr)
    #im_PIL =Image.frombytes('RGB',(bmpinfo['bmWidth'],bmpinfo['bmHeight']),bmpstr)
    box = (8,31,968,511)
    im2 = im_PIL.crop(box)
    #im2.save('./dd2d.jpg')
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hWnd, hWndDC)
    return im2
