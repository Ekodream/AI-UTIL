from pynput import mouse, keyboard
import ctypes

# Set global controllers
mouse_controller = mouse.Controller()
keyboard_controller = keyboard.Controller()

# DPI awareness for Windows
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

def get_screen_size():
    """Return current screen size (width, height) using Win32 GetSystemMetrics.
    """
    try:
        user32 = ctypes.windll.user32
        return (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
    except Exception:
        return (1920, 1080)

# Helper to test modifier keys
def is_modifier(k):
    try:
        return k in (
            keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
            keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
            keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
            keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r
        )
    except Exception:
        return False

# Helper to convert stored key string to keyboard.Key or char
def parse_recorded_key(k):
    # k may be a single-char string like 'a', a quoted char "'v'", or the string of a Key like 'Key.space'
    if not isinstance(k, str):
        return k
    if k.startswith('Key.'):
        try:
            return getattr(keyboard.Key, k.split('.')[-1])
        except Exception:
            return k
    if (len(k) >= 3) and ((k[0] == "'" and k[-1] == "'") or (k[0] == '"' and k[-1] == '"')):
        inner = k[1:-1]
        if len(inner) == 1:
            return inner
        return inner
    if len(k) == 1:
        return k
    return k
