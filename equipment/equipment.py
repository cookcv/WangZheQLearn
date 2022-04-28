from pyminitouch import MNTDevice
import time
from equipment.equipment_listen import AiPlayerListen

class Equipment(AiPlayerListen):
    def __init__(self,device_id,operation_inquire_dict) -> None:
        super().__init__()

        self.add_skill_3='d 0 552 1878 100\nc\nu 0\nc\n'
        self.add_skill_2='d 0 446 1687 100\nc\nu 0\nc\n'
        self.add_skill_1='d 0 241 1559 100\nc\nu 0\nc\n'
        self.buy='d 0 651 207 100\nc\nu 0\nc\n'

        self.equipment = MNTDevice(device_id)
        self.operation_inquire_dict = operation_inquire_dict
        # self.present_operation_dict={"图片号":"0","移动操作":"无移动","动作操作":"无动作"}
        self.present_move_instruction = "移动停"
        

    def send(self,instruction):
        try:
            print("发送指令:",instruction)
            if instruction.startswith("d"):
                self.equipment.connection.send(instruction)
            else:
                self.equipment.connection.send(self.operation_inquire_dict[instruction])
        except:
            self.ai_open = False
            print('发送失败:',instruction)
        time.sleep(0.01)

    def _no_move(self,new_move_instruction) -> bool:
        if new_move_instruction in [self.present_move_instruction, '无移动']:
            return True
        else:
            return False

    def _no_attack(self,new_attack_instruction) -> bool:
        if new_attack_instruction in ['无动作', '发起集合', '发起进攻', '发起撤退']:
            return True
        else:
            return False

    def _change_present_instruction(self,new_instruction):
        present_instruction = new_instruction
        return present_instruction

    def send_move_instruction(self,new_move_instruction):

        if self._no_move(new_move_instruction):
            pass 
        else:
            self.present_move_instruction = self._change_present_instruction(new_move_instruction)
            self.send(self.present_move_instruction)

    def send_atack_instruction(self,new_atack_instruction):
        if self._no_attack(new_atack_instruction):
            pass
        else:
            self.send(new_atack_instruction)

    def buy_add_skill(self):
        ## 该部分可以改成训练获取，目前是定时触发一次
        self.send(self.buy)
        self.send(self.add_skill_3)
        self.send(self.add_skill_2)
        self.send(self.add_skill_1)
        self.send('移动停')
        self.send(self.present_move_instruction)

    

 
    