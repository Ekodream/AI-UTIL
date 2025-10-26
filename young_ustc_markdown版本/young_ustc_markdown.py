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
    """自动查找并点击"下一页"按钮"""
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
import os
import base64
import time
from openai import OpenAI
from datetime import datetime
import webbrowser  # ✅ 新增模块，用于自动打开文件

def analyze_images_with_ai(custom_prompt=None):
    """从 image/ 文件夹读取所有图片并一次性发给 AI 分析，生成 Markdown 表格并自动打开"""
    # 创建markdown输出目录
    md_dir = "markdown"
    os.makedirs(md_dir, exist_ok=True)
    
    client = OpenAI(
        api_key="sk-c25fb29ed49c45d79a237279f41add45",  # ✅ 建议使用环境变量
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 获取所有截图文件
    image_files = sorted([
        f for f in os.listdir("image") if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not image_files:
        print("❌ 没有找到图片，请先运行截图部分。")
        return

    print(f"📸 检测到 {len(image_files)} 张图片，开始批量分析...\n")

    # 构建 AI 输入内容
    content_list = []
    for i, img_name in enumerate(image_files):
        img_path = os.path.join("image", img_name)
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        content_list.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + img_base64}
        })
        print(f"📷 已加载第 {i+1} 张图片：{img_name}")

    # ===============================
    # 让 AI 输出 Markdown 而非 HTML
    # ===============================
    prompt_text = f"""这些图片是中国科学技术大学活动发布平台的截图（共 {len(image_files)} 张）。
请分析每张图片中的活动信息，并生成一份.md 即Markdown格式的表格。

要求如下：
1. 输出格式为纯Markdown表格（| 角标 | 活动名称 | 组织方 | 报名截止时间 |）
2. 表格中严格识别每个活动对应的“德”“智”“体”“美”“劳”角标，严格识别角标!!!严格识别角标!!!
3. 请严格分析各种活动的角标信息，不要全部输出“”
4. 最后请输出一段“共X个活动”的总结
5. 以时间顺序排序
6. 如果存在两个活动时间上可能有冲突(指时间相差小于或等于两小时)，请在最下方将两个活动单独列出来，提醒用户注意时间冲突。
7. 请确保Markdown语法正确，表格对齐美观。
8. 不要包含HTML、CSS或多余说明
"""

    # 支持自定义提示
    if custom_prompt:
        prompt_text += "\n\n用户补充要求：\n" + custom_prompt
        print("🔧 使用自定义提示词。")

    content_list.append({"type": "text", "text": prompt_text})

    print(f"🧠 正在一次性分析 {len(image_files)} 张图片...")

    # 调用 AI 模型
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[{"role": "user", "content": content_list}],
        stream=False,
        extra_body={"enable_thinking": False, "thinking_budget": 81920},
    )

    print("=" * 50 + " 生成 Markdown 文件 " + "=" * 50)

    # 解析 AI 返回内容
    content = completion.choices[0].message.content
    markdown_content = ""
    if isinstance(content, str):
        markdown_content = content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                markdown_content += part["text"]

    # 写入 markdown 文件
    if markdown_content:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = os.path.join(md_dir, f"activities_{timestamp}.md")

        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"✅ Markdown 文件已生成：{md_filename}")

        # ✅ 自动打开Markdown文件
        try:
            print("📖 正在打开文件...")
            if os.name == 'nt':  # Windows
                os.startfile(md_filename)
            elif os.name == 'posix':  # macOS / Linux
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                os.system(f"{opener} '{md_filename}'")
            else:
                webbrowser.open(md_filename)
        except Exception as e:
            print(f"⚠️ 无法自动打开文件：{e}")
            print(f"请手动打开 {md_filename}")
    else:
        print("❌ AI 未返回有效内容")

    print("\n✅ 批量图片分析完成！")

# ============================================
# 三、主程序入口
# ============================================
def main():
    """主程序入口"""
    print("=" * 50)
    print("🎓 中国科学技术大学活动发布平台分析工具")
    print("=" * 50)
    
    while True:
        print("\n请选择操作：")
        print("1. 自动截图并翻页")
        print("2. 使用默认提示词分析图片并生成markdown")
        print("3. 使用自定义提示词分析图片并生成markdown")
        print("4. 退出程序")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == "1":
            print("程序将在 3 秒后开始截图，请切换到需要截图的窗口...")
            time.sleep(3)
            capture_all_pages()
        
        elif choice == "2":
            analyze_images_with_ai()
        
        elif choice == "3":
            print("\n" + "=" * 30)
            print("自定义提示词输入")
            print("=" * 30)
            print("💡 提示：请输入你希望AI如何分析图片的指令")
            print("💡 建议包含：输出格式、分析重点、特殊要求等")
            print("💡 留空并按回车将使用默认提示词")
            print("-" * 30)
            
            custom_prompt = input("请输入自定义提示词: ").strip()
            
            if not custom_prompt:
                print("⚠️ 未输入自定义提示词，将使用默认提示词")
                analyze_images_with_ai()
            else:
                print(f"✅ 已设置自定义提示词（长度: {len(custom_prompt)} 字符）")
                analyze_images_with_ai(custom_prompt)
        
        elif choice == "4":
            print("👋 程序已退出，再见！")
            break
        
        else:
            print("❌ 无效选项，请重新输入")

if __name__ == "__main__":
    main()