"""事件执行模块

负责读取/执行由大模型生成的操作序列（JSON 格式）。
支持的事件类型：
- click: {'type':'click', 'x':int, 'y':int, 'button':'left'|'right', 'clicks':int}
- click_image: {'type':'click_image', 'path':'path/to/image.png', 'confidence':0.8}
- type_text: {'type':'type_text', 'text':'hello', 'interval':0.05}
- hotkey: {'type':'hotkey', 'keys':['ctrl','v']}
- wait: {'type':'wait', 'seconds':1.5}
- move: {'type':'move', 'x':100, 'y':200, 'duration':0.2}
- scroll: {'type':'scroll', 'clicks':-500}

注意：执行真实 GUI 操作有风险，运行前请确保理解动作并在安全环境中测试。
"""

import time
import json
import logging
import os

try:
    import pyautogui
except Exception as e:
    pyautogui = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_events_from_file(path: str):
    """从 JSON 文件加载事件序列"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 支持直接为字典中含有 'events' 键
    if isinstance(data, dict) and 'events' in data:
        return data['events']
    if isinstance(data, list):
        return data
    raise ValueError('不支持的事件文件格式，期待 list 或 包含 events 键的 dict')


def execute_event(ev: dict, dry_run: bool = False):
    """执行单个事件，返回执行结果字典"""
    etype = ev.get('type')
    logging.info(f"执行事件: {etype} -> {ev}")
    if dry_run:
        return {'ok': True, 'event': ev, 'dry_run': True}

    if pyautogui is None:
        raise RuntimeError('缺少 pyautogui，无法执行 GUI 操作。请运行: pip install pyautogui')

    try:
        if etype == 'click':
            x = ev['x']
            y = ev['y']
            button = ev.get('button', 'left')
            clicks = ev.get('clicks', 1)
            duration = ev.get('duration', 0.0)
            pyautogui.moveTo(x, y, duration=duration)
            pyautogui.click(x, y, clicks=clicks, button=button)
            return {'ok': True}

        elif etype == 'click_image':
            image_path = ev['path']
            confidence = ev.get('confidence', 0.8)
            # locateCenterOnScreen 在没有 opencv 时不支持 confidence 参数
            try:
                center = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
            except TypeError:
                center = pyautogui.locateCenterOnScreen(image_path)
            if center is None:
                return {'ok': False, 'error': 'image_not_found', 'path': image_path}
            pyautogui.moveTo(center.x, center.y, duration=0.1)
            pyautogui.click()
            return {'ok': True, 'pos': (center.x, center.y)}

        elif etype == 'type_text':
            text = ev.get('text', '')
            interval = ev.get('interval', 0.05)
            pyautogui.write(text, interval=interval)
            return {'ok': True}

        elif etype == 'hotkey':
            keys = ev.get('keys', [])
            if not keys:
                return {'ok': False, 'error': 'no_keys'}
            pyautogui.hotkey(*keys)
            return {'ok': True}

        elif etype == 'wait':
            seconds = float(ev.get('seconds', 1.0))
            time.sleep(seconds)
            return {'ok': True}

        elif etype == 'move':
            x = ev['x']
            y = ev['y']
            duration = ev.get('duration', 0.1)
            pyautogui.moveTo(x, y, duration=duration)
            return {'ok': True}

        elif etype == 'scroll':
            clicks = int(ev.get('clicks', 0))
            pyautogui.scroll(clicks)
            return {'ok': True}

        else:
            return {'ok': False, 'error': 'unknown_event_type', 'type': etype}

    except Exception as e:
        logging.exception('执行事件时出错')
        return {'ok': False, 'error': str(e)}


def execute_events(events, dry_run: bool = False, stop_on_error: bool = True):
    """执行事件列表，返回每个事件的结果列表"""
    results = []
    for ev in events:
        res = execute_event(ev, dry_run=dry_run)
        results.append(res)
        if not res.get('ok') and stop_on_error:
            logging.error('事件执行失败，停止后续操作: %s', res)
            break
    return results


if __name__ == '__main__':
    # 测试加载与执行（仅用于开发调试）
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='事件 JSON 文件', default='./json/actions.json')
    parser.add_argument('--dry', action='store_true', help='只打印不执行')
    args = parser.parse_args()
    if not os.path.exists(args.file):
        print('示例文件不存在，创建一个 sample.json')
        sample = [
            {'type': 'wait', 'seconds': 1},
            {'type': 'move', 'x': 100, 'y': 100, 'duration': 0.2},
            {'type': 'click', 'x': 100, 'y': 100},
            {'type': 'type_text', 'text': 'hello from events.py'}
        ]
        os.makedirs(os.path.dirname(args.file) or '.', exist_ok=True)
        with open(args.file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print('已写入', args.file)
    events = load_events_from_file(args.file)
    print('Loaded events:', events)
    print('Executing (dry run =', args.dry, ')')
    print(execute_events(events, dry_run=args.dry))
