import sys
import os
import subprocess
from PyQt5.QtWidgets import (
    QWidget, QLabel, QFrame, QVBoxLayout, QHBoxLayout,
    QGridLayout, QScrollArea, QGraphicsDropShadowEffect, 
    QPushButton, QLineEdit
)
from PyQt5.QtCore import Qt, QPoint, QPropertyAnimation, QRect, QTimer, QRectF 
from PyQt5.QtGui import QFont, QColor, QPainter, QLinearGradient, QIcon, QPen
from PyQt5.QtCore import QThread, pyqtSignal


class SearchThread(QThread):
    results_ready = pyqtSignal(list)

    def __init__(self, root, query):
        super().__init__()
        self.root = root
        self.query = query.lower()

    def run(self):
        matches = []
        for root, dirs, files in os.walk(self.root):
            for d in dirs:
                if self.query in d.lower():
                    matches.append(os.path.join(root, d))
            for f in files:
                if self.query in f.lower():
                    matches.append(os.path.join(root, f))
            if len(matches) >= 60:
                break
        self.results_ready.emit(matches[:60])
        
#HUD CARD
class HudCard(QFrame):
    def __init__(self, title: str, subtitle: str = "", footnote: str = "", desktop=None, parent=None):
        super().__init__(parent)
        self.desktop = desktop
        self.title = title
        self.subtitle = subtitle
        self.footnote = footnote
        self._dragging = False
        self._origin_folder = None
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.OpenHandCursor)
        self.setFixedSize(300, 120)
        self._hover = False
        self.setMouseTracking(True)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setOffset(0, 4)
        self.shadow.setColor(QColor(0, 255, 200, 150))
        self.setGraphicsEffect(self.shadow)
    
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(4, 4, -4, -4)

        #Background 
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0, QColor(65, 90, 120, 230))
        grad.setColorAt(1, QColor(30, 50, 75, 210))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(r, 14, 14)
        
        #Hover corners
        if self._hover:
            penL = QPen(QColor(0, 255, 255, 160), 1)
            p.setPen(penL)
            for dx, dy in [(12, 0), (0, 12)]:
                p.drawLine(r.topLeft(), r.topLeft() + QPoint(dx, dy))
                p.drawLine(r.topRight(), r.topRight() + QPoint(-dx, dy))
                p.drawLine(r.bottomLeft(), r.bottomLeft() + QPoint(dx, -dy))
                p.drawLine(r.bottomRight(), r.bottomRight() + QPoint(-dx, -dy))

        #Decorative lines
        p.setPen(QPen(QColor(0, 255, 255, 70), 1))
        p.drawLine(r.left() + 20, r.top() + 28, r.right() - 20, r.top() + 28)
        p.drawLine(r.left() + 16, r.bottom() - 28, r.right() - 16, r.bottom() - 28)

        #Title
        p.setFont(QFont("Arial", 8, QFont.Bold))
        p.setPen(QColor(0, 255, 255, 180))
        p.drawText(r.adjusted(16, 10, -10, -90), Qt.AlignLeft, self.title.upper())

        #Subtitle
        p.setFont(QFont("Consolas", 17, QFont.Bold))
        p.setPen(QColor(255, 255, 255, 235))
        p.drawText(QRectF(r.adjusted(16, 38, -160, -30)), Qt.AlignLeft, self.subtitle)
        
        #Footnote
        if self.footnote:
            p.setFont(QFont("Arial", 7))
            p.setPen(QColor(0, 255, 255, 100))
            p.drawText(r.adjusted(18, r.height() - 42, -14, -10), Qt.AlignLeft, self.footnote)

        p.end()   
    
    def setSubtitle(self, new_text):
        self.subtitle = new_text
        self.update()    
    
    def setFootnote(self, new_text):
        self.footnote = new_text
        self.update()
    
    def enterEvent(self, _):
        self._hover = True
        self.update()

    def leaveEvent(self, _):
        self._hover = False
        self.update()
        
    def _snap_to_grid(self):
        x, y = self.pos().x(), self.pos().y()
        grid_x = round(x / 20) * 20
        grid_y = round(y / 20) * 20
        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(220)
        anim.setEndValue(QRect(grid_x, grid_y, self.width(), self.height()))
        anim.start()
        self.anim = anim    
        
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.raise_()
            if self._origin_folder is not None:
                global_pos = self.mapToGlobal(QPoint(0, 0))
                self.setParent(self.desktop)
                self.move(self.desktop.mapFromGlobal(global_pos))
                self.show()
                if self._origin_folder.popup and self._origin_folder.popup.isVisible():
                    self._origin_folder.popup.close()
                self._origin_folder.release(self)
                self._origin_folder = None
            self.resize(240, 100)
            self._dragging = True
            self._offset = ev.globalPos() - self.mapToGlobal(QPoint(0, 0))
            self.setCursor(Qt.ClosedHandCursor)
            self.raise_()
            
    def mouseMoveEvent(self, ev):
        if self._dragging:
            self.move(self.mapToParent(ev.pos() - self._offset))
            
    def mouseReleaseEvent(self, ev):
        if self._dragging:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
            self.desktop.resolve_drop(self)
            self._snap_to_grid()

class FolderIcon(QFrame):
    def __init__(self, label: str, desktop, parent=None, is_file_folder=False, path=None):
        super().__init__(parent)
        self.desktop = desktop
        self.label = label
        self.contents = []
        self.popup = None
        self._dragging = False
        self._origin_folder = None  
        self._offset = QPoint()
        self.is_file_folder = is_file_folder
        self.path = path
        self.setFixedSize(120, 90)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_TranslucentBackground)
        t = QLabel(label, self)
        t.setFont(QFont("Arial", 9, QFont.Bold))
        t.setStyleSheet("color:#00ffff")
        t.move(10, 12)
        self.badge = QLabel("0", self)
        self.badge.setFont(QFont("Arial", 8, QFont.Bold))
        self.badge.setStyleSheet("color:white; background:#ff006a; border-radius:7px;")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.resize(14, 14)
        self.update_badge()
        
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, QColor(45, 75, 115, 230))
        g.setColorAt(1, QColor(25, 45, 75, 210))
        p.setBrush(g)
        p.setPen(QColor(0, 200, 255, 120))
        p.drawRoundedRect(r.adjusted(0, 0, -1, -1), 13, 13)
        
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            if self._origin_folder is not None:
                global_pos = self.mapToGlobal(QPoint(0, 0))
                self.setParent(self.desktop)
                self.move(self.desktop.mapFromGlobal(global_pos))
                self.show()
                if self._origin_folder.popup and self._origin_folder.popup.isVisible():
                    self._origin_folder.popup.close()
                self._origin_folder.release(self)
                self._origin_folder = None
            self._dragging = True
            self._offset = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.raise_()
        elif ev.button() == Qt.RightButton:
            self.open_popup()
            
    def mouseMoveEvent(self, ev):
        if self._dragging:
            self.move(self.mapToParent(ev.pos() - self._offset))
            
    def mouseReleaseEvent(self, ev):
        self._dragging = False
        self.setCursor(Qt.PointingHandCursor)
        if not self.is_file_folder:
            self.desktop.resolve_folder_drop(self)
            
    def mouseDoubleClickEvent(self, ev):
        self.open_popup()
        
    def update_badge(self):
        n = len(self.contents)
        self.badge.setText(str(n))
        self.badge.move(self.width() - 20, 10)
        self.badge.setVisible((n > 0) and not self.is_file_folder)
        
    def absorb(self, item):
        if item in self.contents:
            return
        item._origin_folder = self  #this works for both HudCard and FolderIcon
        self.contents.append(item)
        self.update_badge()
        if self.popup:
            self.popup.refresh()
            
    def release(self, card):
        if card in self.contents:
            self.contents.remove(card)
            self.update_badge()
            if self.popup:
                self.popup.refresh()
                
    def open_popup(self):
        if self.popup and self.popup.isVisible():
            return
        if self.is_file_folder:
            self.popup = FileFolderPopup(self, self.path)
        else:
            self.popup = FolderPopup(self)
        #self.popup.move(self.mapToGlobal(QPoint(0, self.height() + 8))) #oldline made popup go out of screen
        
        desired_pos = self.mapToGlobal(QPoint(0, self.height() + 8))
        popup_size = self.popup.sizeHint()

        #clamp to parent window boundaries
        parent = self.window()  #QMainWindow or container
        parent_geom = parent.geometry()

        #calculate screen safe position so it doesnt go off the screen 
        clamped_x = min(
            max(desired_pos.x(), parent_geom.left() + 20),
            parent_geom.right() - popup_size.width() - 20
        )
        clamped_y = min(
            max(desired_pos.y(), parent_geom.top() + 20),
            parent_geom.bottom() - popup_size.height() - 20
        )

        self.popup.move(QPoint(clamped_x, clamped_y))
        self.popup.show()


#FILE FOLDER POPUP (Finder like with search and breadcrumbs)
class FileFolderPopup(QWidget):
    def __init__(self, folder_icon: FolderIcon, path, parent=None, breadcrumbs=None):
        super().__init__(parent)
        self.folder_icon = folder_icon
        self.path = os.path.abspath(path)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(740, 540)
        base = QFrame(self)
        base.setGeometry(self.rect())
        base.setStyleSheet("background:rgba(25,35,60,235); border-radius:22px;")
        lay = QVBoxLayout(base)
        lay.setContentsMargins(22, 22, 22, 22)
        self.breadcrumbs = breadcrumbs if breadcrumbs else [self.path]
        
        #Breadcrumb navigation
        bc = QHBoxLayout()
        for i, crumb in enumerate(self.breadcrumbs):
            crumb_btn = QPushButton(os.path.basename(crumb) or crumb)
            crumb_btn.setStyleSheet(
                "QPushButton { color: #00ffff; background:transparent; border:none; font-weight:bold;}"
                "QPushButton:hover {color:white;}"
            )
            crumb_btn.clicked.connect(lambda _, p=crumb: self.go_to_breadcrumb(p))
            bc.addWidget(crumb_btn)
            if i < len(self.breadcrumbs) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet("color:#00d9e7; font-size:15px;")
                bc.addWidget(arrow)
        bc.addStretch()
        lay.addLayout(bc)
        #Search bar
        self.searchbar = QLineEdit()
        self.searchbar.setPlaceholderText("🔎 Search all files/folders...")
        self.searchbar.setStyleSheet(
            "QLineEdit { background:rgba(20,35,60,180); color:#0ff; border: 1px solid #00d9e7; border-radius: 9px; padding: 7px; font-size:13px; }"
        )
        lay.addWidget(self.searchbar)
        self.searchbar.textChanged.connect(self.schedule_search)
        #File/folder area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border:none;background:transparent;")
        self.inner = QWidget()
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setSpacing(8)
        self.scroll.setWidget(self.inner)
        lay.addWidget(self.scroll)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        
    def schedule_search(self):
        self.timer.stop()  # Cancel any queued search
        self.timer.start(220)
        
    def go_to_breadcrumb(self, p):
        idx = self.breadcrumbs.index(p)
        popup = FileFolderPopup(self.folder_icon, p, breadcrumbs=self.breadcrumbs[:idx+1])
        #popup.move(self.pos()) #oldline 
        
        popup.resize(740, 540)  # enforce same size manually

        popup_size = popup.sizeHint()
        desired_pos = self.pos()
        parent = self.window()
        parent_geom = parent.geometry()
        
        clamped_x = min(
            max(desired_pos.x(), parent_geom.left() + 20),
            parent_geom.right() - popup_size.width() - 20
        )
        clamped_y = min(
            max(desired_pos.y(), parent_geom.top() + 20),
            parent_geom.bottom() - popup_size.height() - 20
        )

        popup.move(QPoint(clamped_x, clamped_y))
        popup.show()
        self.close()
        
    def _clear_results(self):
        while self.vbox.count():
            w = self.vbox.takeAt(0).widget()
            if w:
                w.setParent(None)

    def _display_results(self, found):
        if not found:
            lbl = QLabel("No matches found.")
            lbl.setStyleSheet("color:#ff004a; font-weight:bold;")
            self.vbox.addWidget(lbl)
        for f in found:
            btn = self.make_file_button(f)
            self.vbox.addWidget(btn)
        self.inner.adjustSize()

    def _populate_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            err = QLabel("❌ Folder not found!")
            err.setStyleSheet("color:#ff004a; font-weight:bold;")
            self.vbox.addWidget(err)
            return
        files = sorted(os.listdir(folder_path))
        if not files:
            lbl = QLabel("This folder is empty.")
            lbl.setStyleSheet("color:#666;")
            self.vbox.addWidget(lbl)
        for f in files:
            full = os.path.join(folder_path, f)
            btn = self.make_file_button(full)
            self.vbox.addWidget(btn)
        self.inner.adjustSize()    
        
    def refresh(self):
        search = self.searchbar.text().strip()
        self._clear_results()

        # In refresh()
        if hasattr(self, 'search_thread') and self.search_thread.isRunning():
            self.search_thread.quit()
            self.search_thread.wait()

        if search:
            self.search_thread = SearchThread(self.path, search)
            self.search_thread.results_ready.connect(self._display_results)
            self.search_thread.start()
        else:
            self._populate_folder(self.path)
        
    def make_file_button(self, fpath):
        btn = QPushButton(os.path.basename(fpath) or fpath)
        btn.setStyleSheet(
            "QPushButton { color: #eee; background: rgba(0,255,255,22); border: 1px solid #00d9e7; border-radius: 7px; padding: 7px; text-align:left; }"
            "QPushButton:hover { background: rgba(0,255,255,90); color: #0ff; }"
        )
        btn.setCursor(Qt.PointingHandCursor)
        if os.path.isdir(fpath):
            btn.setIcon(QIcon.fromTheme("folder"))
        else:
            btn.setIcon(QIcon.fromTheme("text-x-generic"))
        btn.clicked.connect(lambda _, p=fpath: self.open_file_or_folder(p))
        return btn
    
    def open_file_or_folder(self, fpath):
        if os.path.isdir(fpath):
            new_breadcrumbs = self.breadcrumbs + [fpath]
            popup = FileFolderPopup(self.folder_icon, fpath, breadcrumbs=new_breadcrumbs)
            #popup.move(self.pos() + QPoint(40, 40)) #oldline 
            
            popup.resize(740, 540)  # enforce same size manually

            popup_size = popup.sizeHint()
            desired_pos = self.pos() + QPoint(40, 40)
            parent = self.window()
            parent_geom = parent.geometry()

            clamped_x = min(
                max(desired_pos.x(), parent_geom.left() + 20),
                parent_geom.right() - popup_size.width() - 20
            )
            clamped_y = min(
                max(desired_pos.y(), parent_geom.top() + 20),
                parent_geom.bottom() - popup_size.height() - 20
            )

            popup.move(QPoint(clamped_x, clamped_y))
            popup.show()
        else:
            #open file with default app
            if sys.platform == "darwin":
                subprocess.call(["open", fpath])
            elif sys.platform == "win32":
                os.startfile(fpath)
            elif sys.platform.startswith("linux"):
                subprocess.call(["xdg-open", fpath])
            else:
                print(f"Don't know how to open files on this platform: {sys.platform}")
    
#FOLDER POPUP (HUD CARD HOLDER, supports stacking folders)
class FolderPopup(QWidget):
    def __init__(self, folder_icon: FolderIcon, parent=None):
        super().__init__(parent)
        self.folder_icon = folder_icon
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(400, 380)
        base = QFrame(self)
        base.setGeometry(self.rect())
        base.setStyleSheet("background:rgba(25,35,60,240); border-radius:22px;")
        lay = QVBoxLayout(base)
        lay.setContentsMargins(22, 22, 22, 22)
        title = QLabel(folder_icon.label)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setStyleSheet("color:#00ffff")
        lay.addWidget(title)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border:none;background:transparent;")
        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setSpacing(14)
        self.scroll.setWidget(self.inner)
        lay.addWidget(self.scroll)
        self.refresh()
        
    def refresh(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.setParent(None)
        row = col = 0
        for card in self.folder_icon.contents:
            card.setParent(self.inner)
            card.show()
            self.grid.addWidget(card, row, col)
            col += 1
            if col == 2:
                col, row = 0, row + 1
        #Show folders (drag to stack folders into folders)
        for fol in self.folder_icon.desktop.folders:
            #Don't allow self‑nesting or duplicates
            if fol is self.folder_icon or fol in self.folder_icon.contents:
                continue
            btn = QPushButton(f"[Stack] {fol.label}")
            btn.setStyleSheet(
                "QPushButton { color:#00cfff; background:rgba(0,255,255,16); border:1px solid #00d9e7; border-radius:7px; padding:7px;}"
                "QPushButton:hover {background:rgba(0,255,255,100);color:white;}"
            )
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, f=fol: self.stack_folder_in_folder(f))
            self.grid.addWidget(btn, row, col)
            col += 1
            if col == 2:
                col, row = 0, row + 1
        self.inner.adjustSize()
        
    def stack_folder_in_folder(self, folder):
        if folder not in self.folder_icon.contents:
            self.folder_icon.absorb(folder)
            self.refresh()
