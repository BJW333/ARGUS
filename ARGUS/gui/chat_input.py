# chat_input.py
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTextEdit, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QEvent

class ChatInput(QWidget):
    submitted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setStyleSheet("""
        QTextEdit {
            background: rgba(28,38,58,0.92);
            border: 1px solid rgba(80,170,255,.35);
            border-radius: 10px;
            color: #E7F0FF;
            padding: 8px;
        }
        QPushButton {
            background: rgba(120,170,255,0.16);
            border: 1px solid rgba(120,170,255,0.35);
            border-radius: 8px;
            color: #DDEBFF;
            padding: 6px 14px;
        }
        QPushButton:hover { background: rgba(120,170,255,0.28); }
        """)

        self.edit = QTextEdit(self)
        self.edit.setPlaceholderText("Message ARGUS…  (Enter to send • Shift+Enter for newline)")
        self.edit.installEventFilter(self)
        
        self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit.setFixedHeight(44)          # one-line look; change to 60 if you want 2 lines
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.send_btn = QPushButton("Send", self)
        self.send_btn.clicked.connect(self._send)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.send_btn, 0)

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.KeyPress:
            if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and not (ev.modifiers() & Qt.ShiftModifier):
                self._send()
                return True
        return super().eventFilter(obj, ev)

    def _send(self):
        text = self.edit.toPlainText().strip()
        if text:
            self.submitted.emit(text)
            self.edit.clear()