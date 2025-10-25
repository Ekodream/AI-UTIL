"""主执行脚本：

功能：
- 读取用户自然语言（从命令行参数）
- 截取屏幕并保存
- 假设调用大模型 API（此处用占位函数）将截图 + 自然语言转为动作序列 JSON
- 将 JSON 写入文件并调用 events.execute_events 执行

注：真实集成模型接口需替换 placeholder 的实现。
"""
import os
import sys
import json
import time
import argparse
from screenshoter import Screenshoter
import events


def placeholder_infer_actions_from_image_and_text(image_path: str, user_text: str):
    """占位函数：根据截图与用户描述生成动作序列。
    真正的实现应调用大模型 API（例如 OpenAI 或厂商提供的多模态模型），并解析返回的操作 JSON。
    这里返回一个示例序列用于演示。
    """
    # 简单示例：打开开始菜单、输入文本、回车（注意：坐标为示例）
    return [
        {'type': 'wait', 'seconds': 0.5},
        {'type': 'move', 'x': 50, 'y': 1050, 'duration': 0.2},
        {'type': 'click', 'x': 50, 'y': 1050, 'button': 'left'},
        {'type': 'wait', 'seconds': 0.3},
        {'type': 'type_text', 'text': user_text, 'interval': 0.03},
        {'type': 'wait', 'seconds': 0.2},
        {'type': 'hotkey', 'keys': ['enter']}
    ]


def save_actions(actions, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'events': actions}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', '-t', help='用自然语言描述你想执行的操作', required=True)
    parser.add_argument('--out', '-o', help='动作 JSON 输出路径', default='./json/actions.json')
    parser.add_argument('--dry', action='store_true', help='仅生成并打印动作，不实际执行')
    args = parser.parse_args()

    s = Screenshoter()
    print('正在截取屏幕...')
    img_path = s.take_screenshot_dir(basename='screencap', directory='./json')
    print('截图保存到', img_path)

    print('调用模型（占位）生成动作序列...')
    actions = placeholder_infer_actions_from_image_and_text(img_path, args.text)
    print('生成动作：', actions)

    save_actions(actions, args.out)
    print('已保存动作到', args.out)

    if args.dry:
        print('干运行，退出')
        return

    print('开始执行动作（3 秒后开始，可通过 Ctrl+C 取消）...')
    time.sleep(3)
    results = events.execute_events(actions, dry_run=False)
    print('执行结果：', results)


if __name__ == '__main__':
    main()
