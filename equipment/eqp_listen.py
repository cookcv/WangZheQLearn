from pynput.keyboard import Key, Listener
from pynput import keyboard
import time, threading



def on_release(key):
    global button1,state
    key_name=get_key_name(key)
    if key_name == '1':
        button1 = False
    if key_name == '2':
        button2 = False
    if key_name == '3':
        button3 = False
    if key_name == '4':
        button4 = False
    if key_name == '5':
        button5 = False
    if key_name == '6':
        button6 = False
    if key_name == '7':
        button7 = False
    if key_name == '8':
        button8 = False
    if key_name == 'Key.page_down':
        state='无状态'
    print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False

def on_press(key):
    global button1,state,game_continue
    key_name=get_key_name(key)
    operation=''
    if key_name=='Key.left':
        state='击杀小兵或野怪或推掉塔'
    if key_name == 'Key.down':
        state='击杀敌方英雄'
    if key_name == 'Key.right':
        state='被击塔攻击'
    if key_name == 'Key.up':
        state='被击杀'
    if key_name == 'Key.page_down':
        state='其它'
    if key_name == 'q':
        state='普通'
    if key_name == 'e':
        state='死亡'
    if key_name == 'i':
        game_continue = bool(1 - game_continue)
    print(state)

def direction_process(press_w,press_s,press_a,press_d,press_q):

    if press_q == True:
        return ('移动停')
    elif press_w == True and press_s == False and press_a == False and press_d == False:
        return ('上移')
    elif press_w == False and press_s == True and press_a == False and press_d == False:
        return ('下移')
    elif press_w == False and press_s == False and press_a == True and press_d == False:
        return ('左移')
    elif press_w == False and press_s == False and press_a == False and press_d == True:
        return ('右移')
    elif press_w == True and press_s == False and press_a == True and press_d == False:
        return ('左上移')
    elif press_w == True and press_s == False and press_a == False and press_d == True:
        return ('右上移')
    elif press_w == False and press_s == True and press_a == True and press_d == False:
        return ('左下移')
    elif press_w == False and press_s == True and press_a == False and press_d == True:
        return ('右下移')
    else:
        return ('')
        
state='无状态'
game_continue = True
th = threading.Thread(target=start_listen,)
th.start()
