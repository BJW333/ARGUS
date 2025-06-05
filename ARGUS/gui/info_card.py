from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QPoint, QPropertyAnimation, QRect, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QLinearGradient, QBrush


class CardManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedHeight(50)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(8)
        self.cards = []

    def add_card(self, card: QWidget, label: str):
        btn = QPushButton(label)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 255, 255, 35);
                color: #00ffff;
                border: 1px solid rgba(0,255,255,100);
                border-radius: 10px;
                font: bold 10pt 'Arial';
                padding: 4px 12px;
            }
            QPushButton:hover {
                background-color: rgba(0, 255, 255, 70);
            }
        """)
        btn.clicked.connect(lambda: self._restore_card(card, btn))
        self.layout.addWidget(btn)
        self.cards.append((card, btn))

    def _restore_card(self, card, button):
        card.show()
        card.raise_()
        button.hide()


class HudCard(QWidget):
    def __init__(self, title, subtitle, footnote="", manager=None, parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self.footnote = footnote
        self.manager = manager
        self.setFixedSize(300, 120)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._dragging = False
        self._pinned = False
        self._hover = False
        self.setMouseTracking(True)

    def paintEvent(self, _):
        p = QPainter(self)
        if not p.isActive():
            return
        try:
            p.setRenderHint(QPainter.Antialiasing)
            r = self.rect().adjusted(4, 4, -4, -4)

            grad = QLinearGradient(r.topLeft(), r.bottomRight())
            grad.setColorAt(0, QColor(10, 30, 40, 25))
            grad.setColorAt(1, QColor(0, 120, 255, 18))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(r, 16, 16)

            if self._hover or self._pinned:
                penL = QPen(QColor(0, 255, 255, 160), 1)
                p.setPen(penL)
                br = 12
                p.drawLine(r.topLeft(), r.topLeft() + QPoint(br, 0))
                p.drawLine(r.topLeft(), r.topLeft() + QPoint(0, br))
                p.drawLine(r.topRight(), r.topRight() - QPoint(br, 0))
                p.drawLine(r.topRight(), r.topRight() + QPoint(0, br))
                p.drawLine(r.bottomLeft(), r.bottomLeft() + QPoint(br, 0))
                p.drawLine(r.bottomLeft(), r.bottomLeft() - QPoint(0, br))
                p.drawLine(r.bottomRight(), r.bottomRight() - QPoint(br, 0))
                p.drawLine(r.bottomRight(), r.bottomRight() - QPoint(0, br))

            p.setPen(QPen(QColor(0, 255, 255, 70), 1))
            p.drawLine(r.left() + 20, r.top() + 28, r.right() - 20, r.top() + 28)
            p.drawLine(r.left() + 16, r.bottom() - 28, r.right() - 16, r.bottom() - 28)

            p.setFont(QFont("Arial", 8, QFont.Bold))
            p.setPen(QColor(0, 255, 255, 180))
            p.drawText(QRectF(r.adjusted(16, 10, -10, -90)), Qt.AlignLeft, str(self.title).upper())

            p.setFont(QFont("Consolas", 17, QFont.Bold))
            p.setPen(QColor(255, 255, 255, 235))
            p.drawText(QRectF(r.adjusted(16, 38, -160, -30)), Qt.AlignLeft, str(self.subtitle))

            if self.footnote:
                p.setFont(QFont("Arial", 7))
                p.setPen(QColor(0, 255, 255, 100))
                p.drawText(QRectF(r.adjusted(18, r.height() - 42, -14, -10)), Qt.AlignLeft, str(self.footnote))
        finally:
            p.end()

    def setSubtitle(self, new_text):
        self.subtitle = new_text
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._pinned:
            self._dragging = True
            self._start_pos = event.globalPos() - self.pos()
        elif event.button() == Qt.RightButton:
            self._collapse()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.move(event.globalPos() - self._start_pos)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self._snap_to_grid()

    def mouseDoubleClickEvent(self, event):
        self._collapse()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            self._pinned = not self._pinned
            self.update()

    def enterEvent(self, _):
        self._hover = True
        self.update()

    def leaveEvent(self, _):
        self._hover = False
        self.update()

    def _collapse(self):
        if self.manager:
            self.hide()
            self.manager.add_card(self, self.title)

    def _snap_to_grid(self):
        x, y = self.pos().x(), self.pos().y()
        grid_x = round(x / 20) * 20
        grid_y = round(y / 20) * 20
        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(220)
        anim.setEndValue(QRect(grid_x, grid_y, self.width(), self.height()))
        anim.start()
        self.anim = anim