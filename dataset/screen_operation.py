
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys
from PIL import ImageQt
import win32gui, win32ui, win32con

class ScreenOperation(object):

    def __init__(self,window_name) -> None:
        window = win32gui.FindWindow(0,window_name)
        self.window = window
        app = QApplication(sys.argv)
        self.screen = app.primaryScreen()

    def get_image(self) -> np.array:
        img = self.screen.grabWindow(self.window)
        image = ImageQt.fromqimage(img)
        image_resize = image.resize((960, 480))
        image_np = np.asarray(image_resize)
        return image_np,image_resize
