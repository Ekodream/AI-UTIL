import pyautogui
import cv2
import numpy as np
import time
import os
from PIL import Image

# -----------------------------
# 配置参数
# -----------------------------
SAVE_DIR = "image"            # 截图保存文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

BUTTON_IMAGE = "next_button.png"  # 翻页按钮截图
CONFIDENCE = 0.8                  # 按钮识别相似度
SCREEN_REGION = None              # None = 全屏截图，也可改为 (x, y, w, h)
SIMILARITY_THRESHOLD = 0.99       # 两张截图相似度阈值

# -----------------------------
# 截图函数
# -----------------------------
def take_screenshot(filename):
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    screenshot.save(filename)
    print(f"[+] 截图已保存：{filename}")

# -----------------------------
# 图片对比函数
# -----------------------------
def images_are_same(img1_path, img2_path, threshold=SIMILARITY_THRESHOLD):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1.shape != img2.shape:
        return False

    diff = cv2.absdiff(img1, img2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size / 3
    similarity = 1 - non_zero_count / total_pixels
    return similarity > threshold

# -----------------------------
# 自动找到并点击翻页按钮
# -----------------------------
def find_and_click_next():
    """通过图像识别找到翻页按钮并点击"""
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        button_location = pyautogui.locateOnScreen(BUTTON_IMAGE, confidence=CONFIDENCE)
        if button_location is not None:
            x, y = pyautogui.center(button_location)
            pyautogui.click(x, y)
            print(f"[→] 点击翻页按钮，坐标：({x}, {y})")
            time.sleep(2)  # 等待页面加载
            pyautogui.moveTo(100, 200)
            return True
        else:
            attempt += 1
            print("未找到翻页按钮，重试...")
            time.sleep(1)
    print("⚠️ 翻页按钮找不到！可能已到底。")
    return False

# -----------------------------
# 主循环
# -----------------------------
def main():
    print("开始自动截屏 + 自动翻页流程...")
    index = 0

    # 第一次截图
    prev_img = os.path.join(SAVE_DIR, f"page_{index}.png")
    take_screenshot(prev_img)

    while True:
        # 找到按钮并点击
        if not find_and_click_next():
            break

        index += 1
        curr_img = os.path.join(SAVE_DIR, f"page_{index}.png")
        take_screenshot(curr_img)

        # 检查是否到底（两张截图几乎相同）
        if images_are_same(prev_img, curr_img):
            print("检测到两次截图相同，已到底。程序结束。")
            os.remove(curr_img)  # 删除重复的最后一张
            break
        else:
            prev_img = curr_img

    print(f"✅ 截图完成，共 {index} 页。")

# -----------------------------
# 启动程序
# -----------------------------
if __name__ == "__main__":
    time.sleep(5)  # 给你时间切换到浏览器窗口
    main()