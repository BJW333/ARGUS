from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QPainter, QColor, QPen

class ScanLine(QWidget):
    """
    Horizontal pale-cyan scan-line that bounces up/down.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.y   = 0
        self.dir = 1

        t = QTimer(self)
        t.timeout.connect(self._step)
        t.start(25)

    def _step(self):
        h = self.height()
        self.y += self.dir
        if self.y > h or self.y < 0:
            self.dir *= -1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 200, 255, 50), 1))
        p.drawLine(0, self.y, self.width(), self.y)