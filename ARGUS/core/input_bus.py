# core/input_bus.py
from PyQt5.QtCore import (
    QObject, pyqtSignal, pyqtSlot,
    QCoreApplication, QThread, QMetaObject, Qt, Q_ARG
)
import sys
import threading
from typing import List


class _InputBus(QObject):
    submit    = pyqtSignal(str)  # user messages
    assistant = pyqtSignal(str)  # assistant/status stream

    @pyqtSlot(str)
    def _emit_submit(self, text: str):
        self.submit.emit(text)

    @pyqtSlot(str)
    def _emit_assistant(self, text: str):
        self.assistant.emit(text)


bus = _InputBus()

# --- Buffer assistant text until GUI is ready/connected ---
_assistant_ready = False
_assistant_backlog: List[str] = []
_lock = threading.RLock()


def mark_assistant_ready():
    """Call once AFTER MainWindow connects bus.assistant."""
    global _assistant_ready
    with _lock:
        if _assistant_ready:
            return
        _assistant_ready = True
        backlog = _assistant_backlog[:]
        _assistant_backlog.clear()

    # Flush backlog on the GUI thread
    for msg in backlog:
        _queued_call("_emit_assistant", msg)


def _queued_call(method_name: str, payload: str):
    """
    Always deliver to bus on the GUI thread.
    If there's no QApplication yet, buffer assistant; drop submit.
    """
    app = QCoreApplication.instance()
    if app is None:
        # No Qt event loop yet: buffer assistant messages so they aren't lost.
        if method_name == "_emit_assistant":
            with _lock:
                _assistant_backlog.append(payload)
        return

    gui_thread = app.thread()
    if QThread.currentThread() is gui_thread:
        # Already on GUI thread → direct
        getattr(bus, method_name)(payload)
    else:
        # Any other thread → queued to GUI
        QMetaObject.invokeMethod(
            bus, method_name, Qt.QueuedConnection, Q_ARG(str, payload)
        )


def send(text: str):
    """Thread-safe: enqueue a user message to the GUI."""
    if not text or not str(text).strip():
        return
    _queued_call("_emit_submit", str(text))


# def print_to_gui(text: str):
#     if text is None:
#         return
#     msg = str(text)
#     if not msg.strip():
#         return
#     sys.__stdout__.write(
#         f"[print_to_gui] thread={threading.current_thread().name} "
#         f"bus={id(bus)} text={msg!r}\n"
#     )
#     if not _assistant_ready:
#         with _assistant_lock:
#             _assistant_backlog.append(msg)
#         return
#     _queued_call("_emit_assistant", msg)


def print_to_gui(*objects, sep=" ", end="\n"):
    msg = sep.join("" if o is None else str(o) for o in objects)
    msg = msg.replace("\r", "")

    # ensure a trailing newline unless caller explicitly passed end=""
    if end is not None and end != "" and not msg.endswith("\n"):
        msg += end

    # don't nuke leading newlines; they control formatting
    if not msg.strip():
        return

    try:
        sys.__stdout__.write(
            f"[print_to_gui] thread={threading.current_thread().name} bus={id(bus)} text={msg!r}\n"
        )
    except Exception:
        pass

    with _lock:
        if not _assistant_ready:
            _assistant_backlog.append(msg)
            return
    _queued_call("_emit_assistant", msg)


def debug_bus_id(prefix: str):
    try:
        sys.__stdout__.write(f"[{prefix}] bus id={id(bus)}\n")
    except Exception:
        pass