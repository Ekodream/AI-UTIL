#!/usr/bin/env python3
import os
import re
import json
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image
import io
import numpy as np

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
                               output_file: str = "events.json",
                               model: str = 'qwen3-vl-plus',
                               api_key: str | None = None,
                               base_url: str | None = None,
                               user_text: str | None = None):
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
        "    \"type\": \"mouse\",            // 'mouse' 或 'key'\n"
        "    \"button\": \"Button.left\"    // 按键(键盘或鼠标)\n"
        "    \"position\": [0.5000, 0.5000],         // 鼠标位置(相对位置坐标,范围为0-1)，仅mouse类型必需\n"
        "  }\n"
        "]"
    )

    # Build multimodal user content: image + base instruction + optional user description
    user_message = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": (
            "请根据上面的截图识别需要执行的操作并以事件数组形式返回。\n"
            "严格只返回 JSON 数组，遵守上面的字段说明。\n"
            "如果需要输入文本，请产生 type='key' 的事件，data 字段内包含 'text' 字段。\n"
            "不要输出任何非 JSON 内容。"
        )}
    ]
    if user_text:
        # append the user's natural-language request as a separate text block
        user_message.append({"type": "text", "text": user_text})

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

def call_model_find_coord(image_path: str,
                        #   refer_img_path: str = "",
                                output_file: str = "coord_output.json",
                                model: str = 'qwen3-vl-plus',
                                api_key: str | None = None,
                                base_url: str | None = None,
                                user_text: str | None = None):
    """Call the model to find coordinates of certain object and return the coordinates."""
    client = OpenAI(
        api_key=api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY'),
        base_url=base_url or os.getenv('DASHSCOPE_BASE_URL') or os.getenv('OPENAI_BASE_URL')
    )

    # Support various image inputs: file path (str), PIL.Image.Image, numpy.ndarray, bytes
    def to_data_url(img_input):
        # If already a path string
        if isinstance(img_input, str):
            return image_to_data_url(img_input)
        # If PIL Image
        if isinstance(img_input, Image.Image):
            buf = io.BytesIO()
            img_input.save(buf, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
        # If numpy array (H,W or H,W,3/4)
        if isinstance(img_input, np.ndarray):
            # convert RGB(A) if needed
            arr = img_input
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8)
            # ensure RGB order
            if arr.ndim == 2:
                mode = 'L'
            elif arr.shape[2] == 3:
                mode = 'RGB'
            else:
                mode = 'RGBA'
            pil = Image.fromarray(arr, mode=mode)
            buf = io.BytesIO()
            pil.save(buf, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
        # If raw bytes
        if isinstance(img_input, (bytes, bytearray)):
            try:
                Image.open(io.BytesIO(img_input))
                return 'data:image/png;base64,' + base64.b64encode(bytes(img_input)).decode('ascii')
            except Exception:
                pass
        raise RuntimeError('Unsupported image input type for call_model_find_coord')

    data_url = to_data_url(image_path)
    # refer_data_url = to_data_url(refer_img_path)

    system_prompt = (
        "你是一个返回严格机器可读输出的助手。\n"
        "你会根据输入的图片定位用户请求的目标，并且严格只返回一个 JSON 对象（不要返回任何说明或额外文本）。\n"
        "JSON 对象格式示例： {\"position\": [0.5, 0.5]}。\n"
        "字段说明：position 为 [x, y]，其中x,y 范围是 0-1（相对位置）\n"
    )

    user_message = [
        {"type": "image_url", "image_url": {"url": data_url}},
        # {"type": "image_url", "image_url": {"url": refer_data_url}},
        {"type": "text", "text": (
            "请在上图中找到用户指定的目标并返回位置坐标。\n"
            "严格输出单个 JSON 对象，遵守上面的格式示例。\n"
            "不要输出任何非 JSON 内容。"
        )}
    ]
    if user_text:
        user_message.append({"type": "text", "text": user_text})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        stream=False
    )

    # Extract text content robustly (reuse pattern from other function)
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        try:
            content = resp.choices[0].delta.content
        except Exception:
            content = str(resp)

    if not isinstance(content, str):
        content = str(content)

    # Try to find a JSON object in the output
    m = re.search(r"(\{[\s\S]*\})", content, re.S)
    if not m:
        raise RuntimeError(f'未能从模型输出中提取 JSON 对象。原始输出:\n{content}')
    obj_text = m.group(1)

    try:
        obj = json.loads(obj_text)
    except Exception as e:
        raise RuntimeError(f'解析模型返回 JSON 失败: {e}\n原始输出:\n{content}')

    # Normalize position to relative coordinates (0..1)
    pos = obj.get('position')
    units = obj.get('units', '').lower()
    if not pos or len(pos) < 2:
        raise RuntimeError(f'模型返回的 JSON 中缺少 position 字段或格式不正确: {obj}')

    # convert to floats
    try:
        px = float(pos[0])
        py = float(pos[1])
    except Exception as e:
        raise RuntimeError(f'无法解析 position 坐标为数值: {pos} ({e})')

    # get image size
    try:
        if isinstance(image_path, str):
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        elif isinstance(image_path, Image.Image):
            img_w, img_h = image_path.size
        elif isinstance(image_path, np.ndarray):
            img_h, img_w = image_path.shape[:2]
        elif isinstance(image_path, (bytes, bytearray)):
            with Image.open(io.BytesIO(image_path)) as im:
                img_w, img_h = im.size
        else:
            raise RuntimeError('Unsupported image input type for size extraction')
    except Exception as e:
        raise RuntimeError(f'无法读取图片以获取尺寸: {image_path} ({e})')

    # decide if pixels or normalized
    if units.startswith('pixel') or px > 1 or py > 1:
        # treat as pixel coords
        x_rel = px / img_w if img_w > 0 else 0.0
        y_rel = py / img_h if img_h > 0 else 0.0
    else:
        # already normalized
        x_rel = px
        y_rel = py

    # clamp
    x_rel = max(0.0, min(1.0, x_rel))
    y_rel = max(0.0, min(1.0, y_rel))

    # overwrite object to store normalized coords
    obj['position'] = [x_rel, y_rel]
    obj['units'] = 'normalized'

    # Ensure output dir exists and save the normalized object
    outp = Path(output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    # return normalized coordinates (x, y)
    return (x_rel, y_rel)

def call_model_summarize_pictures(image_paths: list,
                                 output_file: str,
                                 model: str = 'qwen3-vl-plus',
                                 api_key: str | None = None,
                                 base_url: str | None = None,
                                 user_text: str | None = None):
    """Send multiple images to the model and get a txt summary.

    The model is instructed to return a single JSON object, for example:
    {
      "summary": "总体描述...",
      "images": [
        {"path": "img1.png", "caption": "...", "notes": "..."},
        ...
      ]
    }
    """
    client = OpenAI(
        api_key=api_key or os.getenv('DASHSCOPE_API_KEY') or os.getenv('OPENAI_API_KEY'),
        base_url=base_url or os.getenv('DASHSCOPE_BASE_URL') or os.getenv('OPENAI_BASE_URL')
    )

    # build multimodal message with multiple image blocks
    user_content = []
    for p in image_paths:
        try:
            data_url = image_to_data_url(p)
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception as e:
            # include a text note about failed image encoding
            user_content.append({"type": "text", "text": f"FAILED_TO_LOAD_IMAGE: {p} ({e})"})

    instr = (
        "请基于上面的图片集合给出一个JSON对象，严格只返回JSON，不要任何额外文本。\n"
        "JSON示例：{\"summary\": \"总体描述\", \"images\": [{\"path\": \"name.png\", \"caption\": \"一句话描述\"}]}\n"
        "其中 images 数组应与发送的图片顺序对应，path 字段填图片文件名或索引。"
    )
    user_content.append({"type": "text", "text": instr})
    if user_text:
        user_content.append({"type": "text", "text": user_text})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个返回严格机器可读输出的助手。仅返回一个 JSON 对象。"},
            {"role": "user", "content": user_content}
        ],
        stream=False
    )

    # extract content
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        try:
            content = resp.choices[0].delta.content
        except Exception:
            content = str(resp)
    if not isinstance(content, str):
        content = str(content)

    # find first JSON object
    m = re.search(r"(\{[\s\S]*\})", content, re.S)
    if not m:
        raise RuntimeError(f'未能从模型输出中提取 JSON 对象。原始输出:\n{content}')
    obj_text = m.group(1)

    try:
        obj = json.loads(obj_text)
    except Exception as e:
        raise RuntimeError(f'解析模型返回 JSON 失败: {e}\n原始输出:\n{content}')

    # Save to output
    outp = Path(output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return str(outp)