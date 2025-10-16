import psutil
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsDropShadowEffect, QStackedLayout, QSizePolicy, QSplitter,
    QApplication,   
)
from PyQt5.QtGui import QColor
from .animated_canvas import AnimatedCanvas
from .hud_overlay import HudOverlay
from .hud_desktop import FolderStack  
from actions.actions import identifynetworkconnect
from .chat_panel import ChatView, StdoutToChat
from .chat_input import ChatInput
# i believe the below import must remain here due to the loading of the objrecog and code model files loading
from core.orchestrator import handle_multi_intent_user_input #, process_user_input 
#below import handles unifiying both the voice input and typeing input
from core.input_bus import bus, send, mark_assistant_ready #, debug_bus_id 
#debug_bus_id("main_window")

from config_metrics.logging import log_debug #, log_metrics

#functions for cards
def get_cpu_usage():
    usage = psutil.cpu_percent(interval=None)  # non-blocking used to be interval=0.1 sleeps the GUI thread for 100ms every refresh
    #First call may return 0.0; subsequent calls are accurate if you want call it once at startup to “prime” it
    return f"{usage:.1f}%"

def network_status():
    return "Online" if identifynetworkconnect() else "Offline"


class OrchestratorWorker(QThread):
    token = pyqtSignal(str)
    error = pyqtSignal(str)
    done  = pyqtSignal()

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # Prefer streaming via callback if your orchestrator supports it
            try:
                out = handle_multi_intent_user_input(self.prompt, token_callback=self.token.emit)
            except TypeError:
                # Fallback: orchestrator returns a full string
                out = handle_multi_intent_user_input(self.prompt)

            if isinstance(out, str) and out:
                # Stream the returned text too
                CHUNK = 256
                for i in range(0, len(out), CHUNK):
                    if self._stop:
                        break
                    self.token.emit(out[i:i+CHUNK])
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()
                
                
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ARGUS")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setStyleSheet("QMainWindow { background: #0A0F1A; }")  # solid dark background
        self.resize(1280, 800)
        self._dragging = False

        #Central container
        self.central = QWidget(self, objectName="central")
        self.central.setStyleSheet("""
            QWidget#central {
                background: rgba(25, 30, 60, 180);
                border-radius: 28px;
            }
        """)
        self.setCentralWidget(self.central)

        drop_shadow = QGraphicsDropShadowEffect(blurRadius=42, offset=QPoint(0, 7), color=QColor(0, 38, 80, 160))
        self.central.setGraphicsEffect(drop_shadow)

        layout = QHBoxLayout(self.central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        #LEFT PANEL 
        left_panel = QWidget()
        left_vlayout = QVBoxLayout(left_panel)
        left_vlayout.setContentsMargins(0, 0, 0, 0)
        left_vlayout.setSpacing(8)

        
        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("left_splitter")
        splitter.setChildrenCollapsible(False)

        brain_container = QWidget()
        brain_container.setMinimumHeight(440)
        stack_layout = QStackedLayout(brain_container)
        stack_layout.setStackingMode(QStackedLayout.StackAll)

        self.animated_canvas = AnimatedCanvas()
        self.overlay = HudOverlay()
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)

        stack_layout.addWidget(self.animated_canvas)
        stack_layout.addWidget(self.overlay)

        #self.chat = ChatView()
        self.chat = ChatView(bubble_max_width=1100, column_max_width=1280)
        
        chat_pane = QWidget()
        chat_v = QVBoxLayout(chat_pane)
        chat_v.setContentsMargins(0, 0, 0, 0)
        chat_v.setSpacing(6)
        chat_v.addWidget(self.chat, 1)

        self.chat_input = ChatInput()
        chat_v.addWidget(self.chat_input, 0)

        self.chat_input.submitted.connect(send)   # typing -> bus (callable, not .emit) # typing enters the bus
        bus.submit.connect(self._on_user_submit)  # bus -> UI # both typing & voice land here
        
        splitter.addWidget(brain_container)
        splitter.addWidget(chat_pane)
        splitter.setSizes([600, 320])
        splitter.setStretchFactor(0, 3)  # brain
        splitter.setStretchFactor(1, 2)  # chat

        
        left_vlayout.addWidget(splitter)   
             
        # Redirect stdout/stderr to a streaming assistant bubble
        self.stdout_redir = StdoutToChat(self.chat, role="assistant")
        #sys.stdout = self.stdout_redir #old line
        #sys.stderr = self.stdout_redir #old line
        
        bus.moveToThread(QApplication.instance().thread()) # move bus to GUI thread
        
        bus.assistant.connect(self._bus_to_chat, Qt.QueuedConnection)  # bus -> UI
        mark_assistant_ready()  
        
        # sanity ping to gui text box to make sure it responds 
        QTimer.singleShot(2000, lambda: bus.assistant.emit("ARGUS is ready"))
        
        
        layout.addWidget(left_panel, stretch=3)  # Larger stretch for left panel

        #RIGHT PANEL: Folder/Card Area 
        right_container = QWidget(self.central)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.hud_desktop = FolderStack(parent=right_container)
        self.hud_desktop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hud_desktop.setMinimumSize(0, 600)
        right_layout.addWidget(self.hud_desktop)

        layout.addWidget(right_container, stretch=1)  # Smaller stretch to shrink width orginal value 1
        
        #add cards and folders to the desktop
        #the files folder is created automatically
        self.hud_desktop.add_card("CPU LOAD", get_cpu_usage(), pos=QPoint(40, 40))
        self.hud_desktop.add_card("NETWORK", network_status(), pos=QPoint(40, 180))
        self.hud_desktop.add_folder("Cards", QPoint(40, 320))

        #Update loop for cards
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_hud_data)
        self.timer.start(8000)  #Update every 8 seconds
    
    @pyqtSlot(str)
    def _bus_to_chat(self, s: str):
        if not s or not s.strip():
            return
        self.stdout_redir.write(s)   # opens a bubble if needed
                 
  
    def _on_user_submit(self, text: str):
        if not text or not text.strip():
            log_debug("[GUI] empty input, ignoring")
            return

        log_debug(f"[GUI] received: {text}")
        self.chat.add_message("user", text)
        
        # Reset the assistant stream (no bubble until first write)
        self.stdout_redir.new_stream("assistant")

        if getattr(self, "_worker", None) and self._worker.isRunning():
            self._worker.stop(); self._worker.wait()

        self.chat_input.setEnabled(False)

        self._worker = OrchestratorWorker(text)
        self._worker.setParent(self)

        # Stream tokens straight into the chat bubble
        def _append_token(s: str):
            if not s or not s.strip():
                return
            self.stdout_redir.write(s)

        #self._worker.token.connect(self.stdout_redir.write) #old line
        self._worker.token.connect(_append_token)
        self._worker.error.connect(lambda msg: self.stdout_redir.write(f"\n**Error:** {msg}\n"))
        self._worker.done.connect(lambda: self.chat_input.setEnabled(True))
        self._worker.done.connect(self.stdout_redir.finalize)

        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))

        self._worker.start()
        
    
    def refresh_hud_data(self):
        #basiccly everytime u add a card and it needs live data you must put smth here 
        self.hud_desktop.update_card("CPU LOAD", get_cpu_usage())
        self.hud_desktop.update_card("NETWORK", network_status())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.overlay.setGeometry(self.animated_canvas.geometry())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and e.pos().y() < 56:
            self._dragging = True
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging:
            self.move(e.globalPos() - self._drag_pos)
            e.accept()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._dragging = False
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        self.close()

    def closeEvent(self, e):
        try:
            if hasattr(self, "_worker") and self._worker and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait()
        except Exception:
            pass
        self.animated_canvas.cleanup()
        e.accept()
        
