# Class for executing events, reading from a json file and processing them.

import time
import json
from pathlib import Path
from pynput import mouse, keyboard
from .core import get_screen_size, parse_recorded_key, keyboard_controller, mouse_controller

class EventExecutor:
    def __init__(self):
        pass

    def load(self, filename: str | None = None):
        # Determine target file: prefer provided filename, else default to package/json/events.json
        if filename is None:
            pkg_dir = Path(__file__).resolve().parent
            project_root = pkg_dir.parent
            filename = str(project_root / 'json' / 'events.json')
        else:
            p = Path(filename)
            if not p.exists():
                # try relative to project root (package parent)
                pkg_dir = Path(__file__).resolve().parent
                project_root = pkg_dir.parent
                alt = project_root / filename
                if alt.exists():
                    filename = str(alt)
                else:
                    # try project_root/json/events.json
                    alt2 = project_root / 'json' / 'events.json'
                    if alt2.exists():
                        filename = str(alt2)
                    else:
                        raise FileNotFoundError(f"Record file not found: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def execute(self, events, stop_event=None):
        running = True
        while running:
            for e in events:
                if stop_event is not None and stop_event.is_set():
                    running = False
                    break
                
                if e['type'] == 'key':
                    k = e['key']
                    pressed = e['pressed']
                    try:
                        kobj = getattr(keyboard.Key, k.split('.')[-1])
                    except Exception:
                        kobj = k
                    if pressed:
                        keyboard_controller.press(kobj)
                    else:
                        keyboard_controller.release(kobj)
                elif e['type'] == 'combo':
                    mods = e.get('modifiers', [])
                    key_str = e.get('key')
                    mod_objs = []
                    for m in mods:
                        try:
                            mo = getattr(keyboard.Key, m.split('.')[-1])
                        except Exception:
                            mo = None
                        if mo is not None:
                            mod_objs.append(mo)
                            keyboard_controller.press(mo)
                    kobj = parse_recorded_key(key_str)
                    # map control chars back
                    if isinstance(kobj, str) and len(kobj) == 1 and ord(kobj) < 32:
                        code = ord(kobj)
                        if 1 <= code <= 26:
                            kobj = chr(code + 96)
                        else:
                            vk = e.get('vk')
                            if vk:
                                try:
                                    kobj = keyboard.KeyCode.from_vk(vk)
                                except Exception:
                                    pass
                    try:
                        keyboard_controller.press(kobj)
                        keyboard_controller.release(kobj)
                    except Exception:
                        pass
                    for mo in reversed(mod_objs):
                        try:
                            keyboard_controller.release(mo)
                        except Exception:
                            pass
                elif e['type'] == 'mouse':
                    nx, ny = e['position']
                    cw, ch = get_screen_size()
                    x = int(max(0, min(cw - 1, round(nx * cw))))
                    y = int(max(0, min(ch - 1, round(ny * ch))))
                    mouse_controller.position = (x, y)
                    # small delay to ensure OS moved the mouse
                    time.sleep(0.02)
                    try:
                        button = getattr(mouse.Button, e['button'].split('.')[-1])
                    except Exception:
                        button = mouse.Button.left
                    mouse_controller.click(button)
                    
            if stop_event is not None and stop_event.is_set():
                break
