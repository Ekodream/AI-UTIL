# Class for handling screenshot events

from PIL import ImageGrab
import os

class Screenshoter:
    """简单的屏幕捕获工具。

    方法:
    - take_screenshot(filename): 捕获全屏并保存到 filename
    - take_screenshot_dir(basename, directory): 在指定目录按时间戳保存截图并返回路径
    """
    def __init__(self):
        pass

    def take_screenshot(self, filename: str):
        """捕获当前屏幕并保存为 filename（包含路径）。"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        screenshot = ImageGrab.grab()
        screenshot.save(filename)
        return filename

    def take_screenshot_dir(self, basename: str = 'screenshot', directory: str = './json'):
        """在指定目录创建截图文件，返回文件路径。"""
        os.makedirs(directory, exist_ok=True)
        # 使用简单命名，避免覆盖
        i = 0
        while True:
            path = os.path.join(directory, f"{basename}{'' if i==0 else '_' + str(i)}.png")
            if not os.path.exists(path):
                break
            i += 1
        return self.take_screenshot(path)