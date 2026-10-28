"""Pure target selection plus optional Windows relative mouse input."""

import ctypes
import math
import sys


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_int32), ("dy", ctypes.c_int32),
                ("mouseData", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
                ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t)]


class _KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16),
                ("dwFlags", ctypes.c_uint32), ("time", ctypes.c_uint32),
                ("dwExtraInfo", ctypes.c_size_t)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_uint32), ("wParamL", ctypes.c_uint16),
                ("wParamH", ctypes.c_uint16)]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", ctypes.c_uint32), ("data", _InputUnion)]


KEYS = {"right": 0x02, "left": 0x01, "shift": 0x10, "ctrl": 0x11, "alt": 0x12}


def select_target(boxes, origin, radius):
    candidates = [box for box in boxes if math.dist(box.center, origin) <= radius]
    return min(candidates, key=lambda box: math.dist(box.center, origin), default=None)


def movement_delta(target, origin, smoothing=5.0, max_step=25, deadzone=2.0):
    """Return bounded relative counts; these are not calibrated game-view pixels."""
    if target is None or math.dist(target.center, origin) <= deadzone:
        return 0, 0
    dx, dy = (target.center[i] - origin[i] for i in (0, 1))
    result = tuple(max(-max_step, min(max_step, round(value / smoothing)))
                   for value in (dx, dy))
    if result == (0, 0):
        return (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
    return result


class WindowsInput:
    def __init__(self, user32=None):
        if user32 is None:
            if sys.platform != "win32":
                raise RuntimeError("鼠标跟随目前仅支持 Windows")
            user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.api = user32
        user32.SendInput.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Input), ctypes.c_int]
        user32.SendInput.restype = ctypes.c_uint32
        user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        user32.GetAsyncKeyState.restype = ctypes.c_int16
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int

    def pressed(self, key):
        return bool(self.api.GetAsyncKeyState(key) & 0x8000)

    def foreground(self):
        handle = self.api.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(1024)
        self.api.GetWindowTextW(handle, title, len(title))
        return handle, title.value

    def allowed(self, title, key, captured_window):
        handle, current_title = self.foreground()
        return (bool(title) and handle == captured_window and
                title.casefold() in current_title.casefold() and self.pressed(key))

    def move(self, dx, dy):
        if not dx and not dy:
            return
        event = _Input(type=0)
        event.mi = _MouseInput(dx=dx, dy=dy, dwFlags=0x0001)
        if self.api.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input)) != 1:
            raise RuntimeError("鼠标输入发送失败；目标程序可能不接受标准 Windows 鼠标输入")


def prepare_dpi():
    if sys.platform == "win32":
        api = ctypes.WinDLL("user32", use_last_error=True)
        if hasattr(api, "SetProcessDpiAwarenessContext"):
            api.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
            api.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        else:
            api.SetProcessDPIAware()
