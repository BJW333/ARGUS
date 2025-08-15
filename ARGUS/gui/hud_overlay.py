from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QPainter, QColor, QPen
from .argus_color_palette   import EDGE_DIM

class HudOverlay(QWidget):
    """Faint cyan grid + slow scanning arc (blue theme)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.phase = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(40)

    def next_frame(self):
        self.phase = (self.phase + 1) % 600
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # — Vertical & horizontal grid (very faint dim-cyan) —
        p.setPen(QPen(EDGE_DIM, 1, Qt.DashLine))
        for x in range(140, w, 170):
            p.drawLine(x, 0, x, h)
        for y in range(160, h, 120):
            p.drawLine(0, y, w, y)

        # — Rotating sweep arc (bright cyan) —
        sweep_color = QColor(0, 255, 255,  90)
        p.setPen(QPen(sweep_color, 2))
        cx, cy, r = w//2, h//2, 280
        angle = (self.phase * 0.6) % 360
        p.drawArc(cx-r, cy-r, r*2, r*2,
                  int((angle-15)*16), int(30*16))