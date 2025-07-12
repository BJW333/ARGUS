import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint, QSize, QRect, QPropertyAnimation, QParallelAnimationGroup, QEasingCurve, QTimer
from .info_card import HudCard, FolderIcon
from functools import partial
from PyQt5.QtGui import QPainter, QPen
from .argus_color_palette   import EDGE_DIM

#DESKTOP ROOT
class FolderStack(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self.cards = []
        self.folders = []
        self.auto_layout = True
        self._is_dragging = False
        self._animating = set()
        self._needs_relayout = False
        self.MARGIN = 20
        self.SPACING = 20
        self.init_files_folder()
        
    def paintEvent(self, event):
        #this function adds the same grid that is on the left side 
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(EDGE_DIM, 1, Qt.DashLine))

        w, h = self.width(), self.height()

        # — Vertical & horizontal grid (very faint dim-cyan) —
        for x in range(140, w, 170):
            p.drawLine(x, 0, x, h)
        for y in range(160, h, 120):
            p.drawLine(0, y, w, y)
        p.end()
        
    def init_files_folder(self):
        root_path = "/" if sys.platform != "win32" else "C:\\"
        self.add_file_folder("Files", root_path)

    def should_relayout(self, widget):
        if widget in self._animating:
            return False
        if widget._origin_folder is not None:
            return False
        if widget.pos() in [QPoint(0, 0), QPoint(-1, -1)]:
            return False
        return True

    def queue_relayout(self):
        if not self._needs_relayout:
            self._needs_relayout = True
            QTimer.singleShot(0, self._trigger_relayout)

    def _trigger_relayout(self):
        self._needs_relayout = False
        self.relayout_items()

    def restore_to_desktop(self, widget, size: QSize):
        widget.resize(size)
        widget.setParent(self)
        widget.show()
        widget.raise_()

        if isinstance(widget, FolderIcon):
            if widget not in self.folders:
                self.folders.append(widget)
        elif isinstance(widget, HudCard):
            if widget not in self.cards:
                self.cards.append(widget)

        if self.auto_layout and self.should_relayout(widget):
            self.relayout_items(animate_widget=widget)

    def resolve_folder_drop(self, folder):
        if folder.popup and folder.popup.isVisible():
            folder.popup.close()

        folder_center = folder.mapToGlobal(folder.rect().center())
        absorbed = False
        for other in self.folders:
            if other is folder or other.is_file_folder:
                continue
            other_rect = QRect(other.mapToGlobal(QPoint(0, 0)), other.size())
            if other_rect.contains(folder_center):
                self.animate_folder_into_folder(folder, other)
                absorbed = True
                break

        if not absorbed and folder._origin_folder:
            folder._origin_folder.release(folder)
            folder._origin_folder = None
            self.restore_to_desktop(folder, QSize(120, 90))

    def add_card(self, title, subtitle, footnote="", pos=None):
        card = HudCard(title, subtitle, footnote, desktop=self, parent=self)
        self.cards.append(card)
        if self.auto_layout:
            self.queue_relayout()
        else:
            card.move(pos or QPoint(0, 0))
            card.show()
        return card

    def add_folder(self, label, pos=None):
        folder = FolderIcon(label, desktop=self, parent=self)
        self.folders.append(folder)
        if self.auto_layout:
            self.queue_relayout()
        else:
            folder.move(pos or QPoint(0, 0))
            folder.show()
        return folder

    def add_file_folder(self, label, path, pos=None):
        folder = FolderIcon(label, desktop=self, parent=self, is_file_folder=True, path=path)
        self.folders.append(folder)
        if self.auto_layout:
            self.queue_relayout()
        else:
            folder.move(pos or QPoint(0, 0))
            folder.show()
        return folder

    def update_card(self, title, new_subtitle):
        for card in self.cards:
            if card.title == title:
                card.setSubtitle(new_subtitle)

    def relayout_items(self, animate_widget=None):
        spacing = self.SPACING
        margin = self.MARGIN
        container_width = self.width()
        if container_width == 0:
            return

        widgets = [w for w in self.cards + self.folders if w._origin_folder is None and w.isVisible()]
        if not widgets:
            return

        item_width = widgets[0].width() + spacing
        item_height = widgets[0].height() + spacing
        max_columns = max(1, (container_width - margin * 2) // item_width)

        for idx, widget in enumerate(widgets):
            row = idx // max_columns
            col = idx % max_columns
            target_pos = QPoint(col * item_width + margin, row * item_height + margin)

            if widget.pos() == target_pos or widget in self._animating:
                continue

            if widget is animate_widget:
                self._animating.add(widget)
                anim = QPropertyAnimation(widget, b"pos")
                anim.setDuration(300)
                anim.setEasingCurve(QEasingCurve.OutCubic)
                anim.setEndValue(target_pos)
                anim.finished.connect(partial(self._animating.discard, widget))
                anim.start()
                widget.anim = anim
            else:
                widget.move(target_pos)
                widget.show()

    def resolve_drop(self, card):
        card_center = card.mapToGlobal(card.rect().center())
        absorbed = False
        for fol in self.folders:
            fol_rect = QRect(fol.mapToGlobal(QPoint(0, 0)), fol.size())
            if fol_rect.contains(card_center) and not fol.is_file_folder:
                self.animate_into_folder(card, fol)
                absorbed = True
                break

        if not absorbed:
            if card._origin_folder:
                card._origin_folder.release(card)
                card._origin_folder = None
            self.restore_to_desktop(card, QSize(240, 100))

    def animate_into_folder(self, card, folder):
        g = QParallelAnimationGroup()
        move = QPropertyAnimation(card, b"pos")
        move.setDuration(250)
        target = folder.geometry().center() - QPoint(card.width() // 2, card.height() // 2)
        move.setEndValue(target)
        g.addAnimation(move)
        shrink = QPropertyAnimation(card, b"size")
        shrink.setDuration(250)
        shrink.setEndValue(QSize(0, 0))
        g.addAnimation(shrink)
        g.finished.connect(lambda: self.finish_absorb(card, folder))
        g.start()
        card.anim_group = g

    def animate_folder_into_folder(self, folder, target):
        g = QParallelAnimationGroup()
        move = QPropertyAnimation(folder, b"pos")
        move.setDuration(250)
        target_pos = target.geometry().center() - QPoint(folder.width() // 2, folder.height() // 2)
        move.setEndValue(target_pos)
        g.addAnimation(move)
        shrink = QPropertyAnimation(folder, b"size")
        shrink.setDuration(250)
        shrink.setEndValue(QSize(0, 0))
        g.addAnimation(shrink)
        g.finished.connect(lambda: self.finish_folder_absorb(folder, target))
        g.start()
        folder.anim_group = g

    def finish_folder_absorb(self, folder, target):
        folder.hide()
        if folder in self.folders:
            self.folders.remove(folder)
        target.absorb(folder)
        if self.auto_layout:
            self.queue_relayout()

    def finish_absorb(self, card, folder):
        card.hide()
        if card in self.cards:
            self.cards.remove(card)
        folder.absorb(card)
        if self.auto_layout:
            self.queue_relayout()