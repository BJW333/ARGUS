import sys
import psutil
from PyQt5.QtCore import Qt, QPoint, QTimer, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTextEdit,
    QGraphicsDropShadowEffect, QStackedLayout, QSizePolicy
)
from PyQt5.QtGui import QColor, QFont
from .animated_canvas import AnimatedCanvas
from .hud_overlay import HudOverlay
from .hud_desktop import FolderStack  
from utilsfunctions import identifynetworkconnect


#functions for cards
def get_cpu_usage():
    usage = psutil.cpu_percent(interval=0.1)
    return f"{usage:.1f}%"

def network_status():
    return "Online" if identifynetworkconnect() else "Offline"


class TextRedirector(QObject):
    write_signal = pyqtSignal(str)

    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.write_signal.connect(self._write)

    def write(self, text):
        self.write_signal.emit(str(text))

    def _write(self, text):
        self.text_edit.moveCursor(self.text_edit.textCursor().End)
        self.text_edit.insertPlainText(text)
        self.text_edit.moveCursor(self.text_edit.textCursor().End)

    def flush(self):
        pass
    
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

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Menlo", 12))
        self.output_area.setStyleSheet("""
            QTextEdit {
                background: rgba(40, 55, 70, 180);
                color: #e0eefe;
                border-radius: 14px;
                border: 2px solid rgba(80, 170, 255, 0.30);
                padding: 8px;
            }
        """)
        self.output_area.setMinimumHeight(140)

        brain_container = QWidget()
        brain_container.setMinimumHeight(440)
        stack_layout = QStackedLayout(brain_container)
        stack_layout.setStackingMode(QStackedLayout.StackAll)

        self.animated_canvas = AnimatedCanvas()
        self.overlay = HudOverlay()
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)

        stack_layout.addWidget(self.animated_canvas)
        stack_layout.addWidget(self.overlay)

        left_vlayout.addWidget(brain_container, stretch=3)
        left_vlayout.addWidget(self.output_area, stretch=1)

        sys.stdout = TextRedirector(self.output_area)
        sys.stderr = TextRedirector(self.output_area)

        layout.addWidget(left_panel, stretch=2)

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
        self.timer.start(8000)

    def refresh_hud_data(self):
        #basiccly everytime u add a card and it needs live data you must put smth here 
        self.hud_desktop.update_card("CPU LOAD", get_cpu_usage())
        self.hud_desktop.update_card("NETWORK", network_status())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.overlay.setGeometry(self.animated_canvas.geometry())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._dragging:
            self.move(e.globalPos() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._dragging = False

    def mouseDoubleClickEvent(self, e):
        self.close()

    def closeEvent(self, e):
        self.animated_canvas.cleanup()
        e.accept()