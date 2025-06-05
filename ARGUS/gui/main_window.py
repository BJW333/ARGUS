# main_window.py

import sys
from PyQt5.QtCore    import Qt, QPoint, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTextEdit,
    QGraphicsDropShadowEffect, QStackedLayout
)
from PyQt5.QtGui     import QColor, QFont

from .animated_canvas import AnimatedCanvas
from .info_card       import HudCard, CardManager
from .hud_overlay     import HudOverlay
from PyQt5.QtCore     import QObject, pyqtSignal
import psutil
from utilsfunctions import identifynetworkconnect

#funcitons for info cards
def get_cpu_usage():
    usage = psutil.cpu_percent(interval=0.1)
    return f"{usage:.1f}%"


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
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("QMainWindow { background: transparent; }")
        self.resize(1000, 700)
        self.setWindowTitle("ARGUS")

        # Central glass container
        self.central = QWidget(self, objectName="central")
        self.central.setStyleSheet("""
            QWidget#central {
                background: rgba(25, 30, 60, 180);
                border-radius: 28px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(
            blurRadius=42, offset=QPoint(0, 7),
            color=QColor(0, 38, 80, 160)
        )
        self.central.setGraphicsEffect(shadow)
        self.setCentralWidget(self.central)

        main_layout = QHBoxLayout(self.central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # LEFT PANEL
        left_container = QWidget(self.central)
        left_vlayout   = QVBoxLayout(left_container)
        left_vlayout.setContentsMargins(0, 0, 0, 0)
        left_vlayout.setSpacing(8)

        brain_container = QWidget(left_container)
        brain_container.setMinimumHeight(400)
        brain_container.setAttribute(Qt.WA_TranslucentBackground)
        stack_layout = QStackedLayout(brain_container)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setStackingMode(QStackedLayout.StackAll)

        self.animated_canvas = AnimatedCanvas(brain_container)
        stack_layout.addWidget(self.animated_canvas)

        self.overlay = HudOverlay(brain_container)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        stack_layout.addWidget(self.overlay)

        left_vlayout.addWidget(brain_container, stretch=3)

        self.output_area = QTextEdit(left_container)
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
        self.output_area.setMinimumHeight(120)
        left_vlayout.addWidget(self.output_area, stretch=1)

        sys.stdout = TextRedirector(self.output_area)
        sys.stderr = TextRedirector(self.output_area)

        main_layout.addWidget(left_container, stretch=3)

        # CARD MANAGER ONLY
        self.card_manager = CardManager(self.central)
        self.card_manager.move(680, 600)
        self.card_manager.show()

        # FLOATING CARDS
        self.cards = [
            HudCard("CPU LOAD", get_cpu_usage(), "", manager=self.card_manager, parent=self),
            HudCard("NETWORK", identifynetworkconnect(), "", manager=self.card_manager, parent=self)
        ]

        self.cards[0].move(710, 60)
        self.cards[1].move(710, 200)
        for card in self.cards:
            card.show()
            
        #set timer to update and refresh cards
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._refresh_cards)
        self.update_timer.start(10000)  # every 1000 ms = 1 second
        
        
        self._dragging = False
        self._drag_pos = QPoint()
    
    # Refresh cards with CPU and network info
    def _refresh_cards(self):
        self.cards[0].setSubtitle(get_cpu_usage())
        self.cards[1].setSubtitle(identifynetworkconnect())
        
        
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