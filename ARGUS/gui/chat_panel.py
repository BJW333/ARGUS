# chat_panel.py — Centered column, polished bubbles, smooth streaming, hover Copy.
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QTextBrowser,
    QSizePolicy, QPushButton, QApplication, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QEvent
from PyQt5.QtGui import QFont, QClipboard
import re
from PyQt5.QtWidgets import QTextEdit

# ---------- Markdown rendering ----------
def markdown_to_html(md: str) -> str:
    try:
        import markdown2
        return markdown2.markdown(
            md,
            extras=["fenced-code-blocks", "tables", "strike", "smarty", "break-on-newline"]
        )
    except Exception:
        md = re.sub(r"```(\w+)?\n(.*?)\n```", r"<pre><code>\2</code></pre>", md, flags=re.S)
        md = md.replace("\n", "<br>")
        return f"<div>{md}</div>"

# ---------- Single bubble ----------
class ChatMessageBubble(QTextBrowser):
    """Single message bubble that auto-sizes to its content (no inner scrollbars)."""
    def __init__(self, role: str, text: str, max_width: int = 780, parent: QWidget = None):
        super().__init__(parent)
        self._role = role
        self._raw_stream = []
        self._render_pending = False

        # Typography: sans for prose, monospace only in code
        self.setFont(QFont("Helvetica Neue", 13))
        self.document().setDocumentMargin(12)

        self.setOpenExternalLinks(True)
        self.setReadOnly(True)
        self.setFrameStyle(0)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        #self.setMaximumWidth(max_width)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        
        if role == "user":
            bg, border = "rgba(78,102,136,0.95)", "rgba(150,210,255,.45)"
        else:
            bg, border = "rgba(30,40,64,0.95)", "rgba(90,170,255,.35)"

        self.setStyleSheet(f"""
            QTextBrowser {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 14px;
                color: #EAF2FF;
                padding: 12px 14px;
                selection-background-color: rgba(120,170,255,0.30);
            }}
            pre {{
                background: rgba(0,0,0,.32);
                border-radius: 10px;
                padding: 10px;
                font-family: Menlo, Consolas, monospace;
                font-size: 12.5px;
                white-space: pre-wrap;
                line-height: 1.35em;
            }}
            code {{ font-family: Menlo, Consolas, monospace; font-size: 12.5px; }}
            table {{ border-collapse: collapse; margin: 8px 0; }}
            td, th {{ border: 1px solid rgba(255,255,255,.15); padding: 6px 8px; }}
            a {{ color:#8cc2ff; text-decoration:none; }}
            a:hover {{ text-decoration: underline; }}
        """)

        # keep text width synced to viewport for correct wrapping
        self.document().setTextWidth(self.viewport().width())
        self.document().contentsChanged.connect(self._autosize)
        QTimer.singleShot(0, self._autosize)

        self.set_markdown(text)

    # ——— markdown helpers ———
    def set_markdown(self, md: str):
        self.setHtml(markdown_to_html(md))

    def append_markdown(self, md_fragment: str):
        html = markdown_to_html(md_fragment)
        cur = self.textCursor()
        cur.movePosition(cur.End)
        cur.insertHtml(html)
        self._autosize()

    # buffered streaming for smooth updates
    def append_stream(self, text: str):
        self._raw_stream.append(text)
        if not self._render_pending:
            self._render_pending = True
            #QTimer.singleShot(40, self._render_stream)
            QTimer.singleShot(60, self._render_stream)

    def _render_stream(self):
        self._render_pending = False
        raw = "".join(self._raw_stream)
        self.setHtml(markdown_to_html(raw))
        self._autosize()

    def finalize_stream(self):
        if self._raw_stream:
            self._render_stream()
        self._raw_stream.clear()   # free memory

    # ——— auto-size to content, no inner scrollbars ———
    def resizeEvent(self, e):
        super().resizeEvent(e)
         # ensure wrapping uses the current viewport width
        self.document().setTextWidth(self.viewport().width())
        QTimer.singleShot(0, self._autosize)
      
    def _autosize(self):
        # compute height from document size + contents margins
        doc_h = self.document().size().height()
        m = self.contentsMargins()
        h = int(doc_h + m.top() + m.bottom() + 2)
        # lock height so the outer scroll area handles scrolling
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)

    def text_markdown(self) -> str:
        return self.toPlainText()

# ---------- Message wrapper (alignment + tools) ----------
class ChatMessageWidget(QWidget):
    def __init__(self, role: str, text: str, bubble_max_width: int = 780, parent: QWidget = None):
        super().__init__(parent)
        self.role = role
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.bubble = ChatMessageBubble(role, text, bubble_max_width, self)

        # Copy button (hidden until hover)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setVisible(False)
        self.copy_btn.setCursor(Qt.PointingHandCursor)
        self.copy_btn.setFixedHeight(24)
        self.copy_btn.setStyleSheet("""
            QPushButton {{
                background: rgba(120,170,255,0.14);
                border: 1px solid rgba(120,170,255,0.32);
                border-radius: 6px;
                color: #DDEBFF;
                padding: 2px 8px;
            }}
            QPushButton:hover {{ background: rgba(120,170,255,0.24); }}
            QPushButton:pressed {{ background: rgba(120,170,255,0.34); }}
        """)
        self.copy_btn.clicked.connect(self._copy_to_clipboard)

        # Align like ChatGPT
        if role == "user":
            row.addStretch(1)
            row.addWidget(self.bubble)
        else:
            row.addWidget(self.bubble)
            row.addStretch(1)

        tools = QHBoxLayout()
        tools.setContentsMargins(6, 0, 6, 0)
        if role == "user":
            tools.addStretch(1)
            tools.addWidget(self.copy_btn, 0, Qt.AlignRight)
        else:
            tools.addWidget(self.copy_btn, 0, Qt.AlignLeft)
            tools.addStretch(1)

        outer.addLayout(row)
        outer.addLayout(tools)

    # hover show/hide tools
    def enterEvent(self, e: QEvent):
        self.copy_btn.setVisible(True)
        return super().enterEvent(e)

    def leaveEvent(self, e: QEvent):
        self.copy_btn.setVisible(False)
        return super().leaveEvent(e)

    # expose ops
    def set_markdown(self, md: str): self.bubble.set_markdown(md)
    def append_markdown(self, md: str): self.bubble.append_markdown(md)
    def append_stream(self, text: str): self.bubble.append_stream(text)
    def finalize_stream(self): self.bubble.finalize_stream()

    def _copy_to_clipboard(self):
        QApplication.clipboard().setText(self.bubble.text_markdown(), QClipboard.Clipboard)

# ---------- Chat view (centered column like ChatGPT) ----------
class ChatView(QWidget):
    def __init__(self, parent: QWidget = None, bubble_max_width: int = 780, column_max_width: int = 860):
        super().__init__(parent)
        self._bubble_max_width = bubble_max_width
        self._last_msg = None

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._pin_threshold_px = 60  # how close to bottom counts as "at bottom"
        self._auto_pin = True        # start pinned

        sb = self.scroll.verticalScrollBar()
        sb.rangeChanged.connect(lambda *_: self._maybe_pin_to_bottom())
        sb.valueChanged.connect(self._on_scroll_value_changed)
        
        # full-width transparent container
        self.container = QWidget()
        self.container.setObjectName("chat_container")
        self.container.setAttribute(Qt.WA_StyledBackground, True)
        self.container.setStyleSheet("#chat_container { background: transparent; }")

        # centered column
        self.column = QWidget(self.container)
        self.column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.column.setObjectName("chat_column")
        self.column.setMaximumWidth(column_max_width)
        self.column.setStyleSheet("#chat_column { background: transparent; }")
        self.column.installEventFilter(self)
        
        center = QHBoxLayout(self.container)
        center.setContentsMargins(12, 8, 12, 8)
        center.addStretch(1)
        center.addWidget(self.column, 1)
        center.addStretch(1)

        self.v = QVBoxLayout(self.column)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(10)
        self.v.addStretch(1)

        self.scroll.setWidget(self.container)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.scroll)

    def _apply_target_width(self):
        col_w = max(0, self.column.width())
        target = max(560, int(col_w * 0.96))   # fill most of the column
        for i in range(self.v.count() - 1):    # skip trailing stretch
            item = self.v.itemAt(i)
            w = item.widget()
            if isinstance(w, ChatMessageWidget):
                w.bubble.setFixedWidth(target)  # <-- guarantees a wide bubble
                
    def _is_near_bottom(self) -> bool:
        sb = self.scroll.verticalScrollBar()
        return (sb.maximum() - sb.value()) <= self._pin_threshold_px

    def _on_scroll_value_changed(self, _value: int):
        # User scrolled: update pin state
        self._auto_pin = self._is_near_bottom()

    def _maybe_pin_to_bottom(self):
        if self._auto_pin:
            self._scroll_to_bottom()
            
    def _scroll_to_bottom(self):
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

    def add_message(self, role: str, text: str) -> 'ChatMessageWidget':
        w = ChatMessageWidget(role, text, self._bubble_max_width, self.column)
        self.v.insertWidget(self.v.count() - 1, w, 0)
        self._last_msg = w
        self._apply_target_width()  # ensure new bubble is wide immediately immediate pass (in case layout is already stable)
        self._maybe_pin_to_bottom()
        QTimer.singleShot(0, self._apply_target_width)  # deferred pass after layout settles
        return w

    def new_stream(self, role: str = "assistant"):
        self._last_msg = self.add_message(role, "")

    def eventFilter(self, obj, ev):
        if obj is self.column and ev.type() == QEvent.Resize:
            # After layout settles, apply the target width
            QTimer.singleShot(0, self._apply_target_width)
        return super().eventFilter(obj, ev)
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._apply_target_width)
        
        
# ---------- Stdout redirection (streaming) ----------
class StdoutToChat(QObject):
    write_signal = pyqtSignal(str)

    def __init__(self, chat_view: ChatView, role: str = "assistant"):
        super().__init__()
        self.chat = chat_view
        self.role = role
        self.current = None
        self._has_written = False
        self.write_signal.connect(self._write)

    def new_stream(self, role: str = None):
        # Do NOT create a bubble yet — just reset state
        if role:
            self.role = role
        self.current = None
        self._has_written = False

    def _ensure_current(self):
        if self.current is None:
            self.current = self.chat.add_message(self.role, "")

    def write(self, text: str):
        self.write_signal.emit(str(text))

    def _write(self, text: str):
        # ignore blanks/keepalives
        if not text or not text.strip():
            return
        self._ensure_current()
        self._has_written = True
        if hasattr(self.current, "append_stream"):
            self.current.append_stream(text)
        else:
            self.current.append_markdown(text)
        QTimer.singleShot(0, self.chat._maybe_pin_to_bottom)

    def finalize(self):
        # Only finalize if something was actually written
        if self.current and self._has_written and hasattr(self.current, "finalize_stream"):
            self.current.finalize_stream()
        self.current = None
        self._has_written = False

    def flush(self):
        pass