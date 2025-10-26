import pyautogui
import cv2
import numpy as np
import time
import os
import base64
from PIL import Image
from openai import OpenAI

# ============================================
# 一、截图与翻页部分
# ============================================
SAVE_DIR = "image"                     # 截图保存文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

BUTTON_IMAGE = "next_button.png"       # 翻页按钮截图文件
CONFIDENCE = 0.8                       # 按钮识别相似度
SCREEN_REGION = None                   # 截屏范围（None = 全屏）
SIMILARITY_THRESHOLD = 0.99            # 判断截图相似度（用于检测到底）

def take_screenshot(filename):
    """截屏保存"""
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    screenshot.save(filename)
    print(f"[+] 截图已保存：{filename}")

def images_are_same(img1_path, img2_path, threshold=SIMILARITY_THRESHOLD):
    """判断两张截图是否相同"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return False
    diff = cv2.absdiff(img1, img2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size / 3
    similarity = 1 - non_zero_count / total_pixels
    return similarity > threshold

def find_and_click_next():
    """自动查找并点击“下一页”按钮"""
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
            pyautogui.click()
            return True
        else:
            attempt += 1
            print("未找到翻页按钮，重试...")
            time.sleep(1)
    print("⚠️ 翻页按钮找不到，可能已到底。")
    return False

def capture_all_pages():
    """自动截屏并保存到 image/ 文件夹"""
    print("开始自动截屏 + 自动翻页流程...")
    index = 0
    prev_img = os.path.join(SAVE_DIR, f"page_{index}.png")
    take_screenshot(prev_img)

    while True:
        if not find_and_click_next():
            break
        index += 1
        curr_img = os.path.join(SAVE_DIR, f"page_{index}.png")
        take_screenshot(curr_img)

        if images_are_same(prev_img, curr_img):
            print("检测到两张截图相同，已到底。程序结束。")
            os.remove(curr_img)
            break
        else:
            prev_img = curr_img

    print(f"✅ 截图完成，共 {index+1} 页。")


# ============================================
# 二、AI图片分析部分
# ============================================
def analyze_images_with_ai():
    """从 image/ 文件夹读取所有图片并一次性发给 AI 分析"""
    client = OpenAI(
        api_key="sk-e80d1c4eec44443291dcc5191271d5c1",  # ⚠️ 请替换为你自己的 API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    image_files = sorted([f for f in os.listdir(SAVE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print("❌ 没有找到图片，请先运行截图部分。")
        return

    print(f"📸 检测到 {len(image_files)} 张图片，开始批量分析...\n")

    # 构建消息内容，包含所有图片
    content_list = []
    
    # 添加所有图片
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(SAVE_DIR, img_name)
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        content_list.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + img_base64}
        })
        print(f"📷 已加载第 {i+1} 张图片：{img_name}")

    # 添加文本提示
    content_list.append({
        "type": "text",
        "text": f"这是2025年中国科学技术大学活动发布平台的界面，共{len(image_files)}张图片。\
            请总结图片中的活动，并以表格的形式列出。\
            格式要求：\
            1.格式：| 角标 | 活动名称 | 组织方 | 报名截止时间 | \
            2.同时将每个活动旁的“德”“智”“体”“美”“劳”五种角标体现在表格中\
            请以表格的形式总结所有图片中的活动信息，将所有活动合并在一个完整的表格中，不需要任何其他输出。"
    })

    print(f"🧠 正在一次性分析 {len(image_files)} 张图片...")
    
    completion = client.chat.completions.create(
        model="qwen3-vl-flash",
        messages=[
            {
                "role": "user",
                "content": content_list
            }
        ],
        stream=False,
        extra_body={
            "enable_thinking": False,
            "thinking_budget": 81920
        },
    )

    print("=" * 50 + " 分析结果 " + "=" * 50)

    # 兼容 Qwen 返回格式（可能是 str 或 list）
    content = completion.choices[0].message.content
    if isinstance(content, str):
        print(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                print(part["text"])
    else:
        print("（无输出）")

    print("\n✅ 批量图片分析完成！")



# ============================================
# 三、主程序入口
# ============================================
if __name__ == "__main__":
    # print("程序将在 5 秒后开始，请切换到需要截图的窗口...")
    # time.sleep(5)

    # # Step 1: 自动截图并翻页
    # capture_all_pages()

    # Step 2: AI 批量识图分析
    analyze_images_with_ai()