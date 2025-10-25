import os
import pyautogui
import numpy as np
import time
from aiutil import screenshoter, callmodel

class YoungUstcUtil:
    def __init__(self):
        self.IMAGE_DIR = "image"            # 截图保存文件夹
        os.makedirs(self.IMAGE_DIR, exist_ok=True)

        self.SAVE_DIR = "Image"
        self.screenshoterobj = screenshoter.Screenshoter()
        self.next_button_pos = (0, 0)
        self.image_dir = os.path.abspath("D:/ToyProjects/AI-UTIL/image")
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_list = []
        self.json_dir = os.path.abspath("D:/ToyProjects/AI-UTIL/json")
        os.makedirs(self.json_dir, exist_ok=True)
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
  
    def screenshot(self, filename: str):
        # save to image_dir with given filename
        path = os.path.join(self.image_dir, filename)
        # Screenshoter has take_screenshot(filename) which accepts full path
        self.screenshoterobj.take_screenshot(path)
        return path

    def find_next_button(self) -> bool:
        img_name = "page_0.png"
        img_path = self.screenshot(img_name)
        self.image_list.append(img_path)
        outfile = os.path.join(self.json_dir, "coord_next_button.json")
        try:
            button_img_path = os.path.join(self.image_dir, "next_button.png")
            coords = callmodel.call_model_find_coord(img_path, outfile, base_url=self.url, user_text='\
                请找到网页中‘下一页’按钮的位置')
        except Exception as e:
            print(f"调用模型定位按钮失败: {e}")
            return False

        # expect coords to be tuple (x_rel, y_rel)
        if not coords or not isinstance(coords, (list, tuple)):
            print("模型未返回有效坐标。")
            return False

        self.next_button_pos = coords
        print(f"找到下一页按钮（相对坐标）: {self.next_button_pos}")
        return True

    def run(self):
        time.sleep(3) 

        if not self.find_next_button():
            print("Next button not found!")
            return
         
        print("开始自动截屏 + 自动翻页流程...")

        index = 0
        for i in range(1):
            index += 1
            img_name = f"page_{index}.png"
            img_path = self.screenshot(img_name)
            self.image_list.append(img_path)
            # 如果已知下一页按钮相对坐标 (x_rel, y_rel)，将其转换为屏幕像素并点击
            if self.next_button_pos and isinstance(self.next_button_pos, (list, tuple)) and len(self.next_button_pos) >= 2:
                try:
                    x_rel, y_rel = float(self.next_button_pos[0]), float(self.next_button_pos[1])
                    sw, sh = pyautogui.size()
                    x = int(max(0, min(sw - 1, round(x_rel * sw))))
                    y = int(max(0, min(sh - 1, round(y_rel * sh))))
                    print(f"点击下一页按钮像素坐标: ({x}, {y})")
                    pyautogui.moveTo(x, y, duration=0.15)
                    pyautogui.click()
                    time.sleep(3.0)  # 等待页面响应
                except Exception as e:
                    print(f"点击下一页按钮失败")
                    time.sleep(3.0)
            else:
                print("没有可用的下一页按钮坐标")
                time.sleep(3.0)

        out_summary = os.path.join(self.json_dir, 'summarize.json')
        try:
            summary_file = callmodel.call_model_summarize_pictures(self.image_list, out_summary, base_url=self.url, user_text='请为这些截图生成总结与每张图片的一句话描述')
            print(f"汇总已保存：{summary_file}")
        except Exception as e:
            print(f"调用模型汇总图片失败: {e}")