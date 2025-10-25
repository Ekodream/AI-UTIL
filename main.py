#!/usr/bin/env python3
import os
import re
import json
import base64
import argparse
from pathlib import Path
from openai import OpenAI
from screenshoter import Screenshoter


def image_to_data_url(path: str) -> str:
    with open(path, 'rb') as f:
        b = f.read()
    return "data:image/png;base64," + base64.b64encode(b).decode('ascii')


def extract_first_json_array(text: str) -> str:
    # Try to find a JSON array in the model output
    m = re.search(r"(\[.*\])", text, re.S)
    if m:
        return m.group(1)
    # fallback: try triple-backtick block
    m2 = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    if m2:
        return m2.group(1)
    raise ValueError('No JSON array found in model output')


def call_model_generate_events(image_path: str,
                               output_file: str,
                               model: str = 'qwen3-vl-plus',
                               api_key: str | None = None,
                               base_url: str | None = None):
    """Call the large model with the screenshot (as data URL) and save returned events JSON to output_file.

    The model is instructed to return only a JSON array describing the sequence of actions.
    """
    client = OpenAI(
        api_key=api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY'),
        base_url=base_url or os.getenv('DASHSCOPE_BASE_URL') or os.getenv('OPENAI_BASE_URL')
    )

    data_url = image_to_data_url(image_path)

    system_prompt = (
        "你是一个返回严格机器可读输出的助手。\n"
        "仅返回一个JSON数组，不要包含任何额外的说明、引导词或格式化文本。\n"
        "数组中的每个元素代表一个操作事件，格式如下(示例)：\n"
        "[\n"
        "  {\n"
        "    \"type\": \"mouse\",            // 'mouse' 或 'key' 或 'combo'\n"
        "    \"position\": [0.5, 0.5],         // 鼠标位置（相对屏幕，0-1），mouse类型必需\n"
        "    \"button\": \"Button.left\"    // 鼠标按键，mouse类型可选\n"
        "  }\n"
        "]"
    )

    user_message = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": (
            "请根据上面的截图识别需要执行的操作并以事件数组形式返回。\n"
            "严格只返回 JSON 数组，遵守上面的字段说明。\n"
            "如果需要输入文本，请产生 type='key' 的事件，data 字段内包含 'text' 字段，或使用 'combo' 表示修饰键组合。\n"
            "不要输出任何非 JSON 内容。"
        )}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        stream=False
    )

    # Extract text content robustly
    content = None
    try:
        # new-style response
        content = resp.choices[0].message.content
    except Exception:
        try:
            content = resp.choices[0].delta.content
        except Exception:
            content = str(resp)

    if not isinstance(content, str):
        content = str(content)

    try:
        arr_text = extract_first_json_array(content)
        events = json.loads(arr_text)
    except Exception as e:
        raise RuntimeError(f'解析模型返回 JSON 失败: {e}\n原始输出:\n{content}')

    # Ensure output dir exists
    outp = Path(output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    return str(outp)


def main():
    p = argparse.ArgumentParser(description='从屏幕截图调用大模型生成操作序列 JSON')
    p.add_argument('--out', '-o', default='./json/events.json', help='输出 JSON 文件路径')
    p.add_argument('--model', default='qwen3-vl-plus', help='模型名称')
    p.add_argument('--api-key', default=None, help='API Key，默认使用环境变量')
    p.add_argument('--base-url', default=None, help='API Base URL，可选')
    p.add_argument('--screenshot-dir', default='./json', help='截图保存目录')
    args = p.parse_args()

    shooter = Screenshoter()
    img_path = shooter.take_screenshot_dir(basename='screenshot', directory=args.screenshot_dir)
    print('截图保存到:', img_path)

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    out_path = call_model_generate_events(img_path, args.out, model=args.model, api_key=args.api_key, base_url=url)
    print('事件已保存到:', out_path)


if __name__ == '__main__':
    main()