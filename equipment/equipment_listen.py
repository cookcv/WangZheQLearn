
from pynput.keyboard import Key, Listener
from pynput import keyboard
import threading

class KeyListenBase:
    def __init__(self) -> None:
        self.lock=threading.Lock()
        self.state = str()

    def _get_key_name(self,key):
        if isinstance(key, keyboard.KeyCode):
            return key.char
        else:
            return str(key)

    # 开始监听
    def start_listen(self):
        with Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _on_press(self,key):
        pass

    def _on_release(self,key):
        pass

class LabelRectifyListen(KeyListenBase):
    def __init__(self) -> None:
        super().__init__()
        self.key_state = {
            'Key.left':'击杀小兵或野怪或推掉塔','Key.down':'击杀敌方英雄',
            'Key.right':'被击塔攻击','Key.up':'被击杀',
            'q':'普通','e':'死亡','p':'过','m':'弃'
            }
    def _on_press(self, key):
        key_name=self._get_key_name(key)
        
    def _on_release(self, key):
        key_name=self._get_key_name(key)
        if key_name in list(self.key_state.keys()):
            self.state = self.key_state[key_name]
        if key == Key.esc:
            # 停止监听
            return False

class AiPlayerListen(KeyListenBase):
    def __init__(self) -> None:
        super().__init__()
        self.manual_operation_list=list()
        self.press_w=False
        self.press_s=False
        self.press_a=False
        self.press_d=False
        self.press_q=False
        self.attack_state=False
        self.ai_open=bool
        self.key_operation = {
            'Key.space':'召唤师技能',
            'Key.end':'补刀',
            'Key.page_down':'推塔',
            'Key.up':'攻击',
            'j':'一技能',
            'k':'二技能',
            'l':'三技能',
            'f':'回城',
            'g':'恢复',
            'h':'召唤师技能',
        }

    def _on_press(self, key):
       
        key_name=self._get_key_name(key)
        operation=''
        if key_name=='w':
            self.press_w=True
        elif key_name=='a':
            self.press_a=True
        elif key_name=='s':
            self.press_s=True
        elif key_name=='d':
            self.press_d=True
        elif key_name == 'q':
            self.press_q=True
        elif key_name == 'i':
            self.ai_open = bool(1 - self.ai_open)
        if key_name in list(self.key_operation.keys()):
            operation = self.key_operation[key_name]
        self.lock.acquire()
        if operation!='':
            self.manual_operation_list.append(operation)
        self.lock.release()

    def _on_release(self, key):
        key_name=self._get_key_name(key)
        if key_name=='w':
            self.press_w=False
        elif key_name=='a':
            self.press_a=False
        elif key_name=='s':
            self.press_s=False
        elif key_name=='d':
            self.press_d=False
        elif key_name == 'q':
            self.press_q = False
        elif key_name=='Key.up' :
            self.lock.acquire()
            self.manual_operation_list.append("无动作")
            self.lock.release()
            # self.attack_state=False
        print("已经释放:", key_name)
        if key == Key.esc:
            # 停止监听
            return False

class StaleLabelListen(KeyListenBase):
    def __init__(self) -> None:
        super().__init__()
        self.game_continue=bool()
        self.key_state = {
            'Key.left':'击杀小兵或野怪或推掉塔','Key.down':'击杀敌方英雄',
            'Key.right':'被击塔攻击','Key.up':'被击杀','p':'其它',
            'q':'普通','e':'死亡','i':bool(1 - self.game_continue)}

    def on_release(self,key):
        key_name=self._get_key_name(key)
        if key_name == 'Key.page_down':
            self.state='无状态'
        print("已经释放:", key_name)
        if key == Key.esc:
            # 停止监听
            return False

    def on_press(self,key):
        key_name=self._get_key_name(key)
        if key_name in list(self.key_state.keys()):
            self.state = self.key_state[key_name]
            print(self.state)


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
        