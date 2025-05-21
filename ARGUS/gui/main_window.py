import sys
from PyQt5.QtCore import Qt, QMetaObject, Q_ARG
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QSplitter

from gui.animated_canvas import AnimatedCanvas, stream



#Output Redirection
class OutputRedirector:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        
        # QMetaObject.invokeMethod(
        #     self.text_edit,
        #     "append",
        #     Qt.QueuedConnection,
        #     Q_ARG(str, text)
        # )
        QMetaObject.invokeMethod(
            self.text_edit,
            "insertPlainText",
            Qt.QueuedConnection,
            Q_ARG(str, text)
        )
        
    def flush(self):
        pass
    
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ARGUS")
        self.resize(600, 750) #orginal values 600 and 600
        
        #Use a splitter to separate the canvas and output area
        splitter = QSplitter(Qt.Vertical)
        
        #your animated canvas (ARGUS visualization)
        self.canvas = AnimatedCanvas(self)
        splitter.addWidget(self.canvas)
        
        #create a text area for terminal output
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #ffffff;
                font-family: Menlo, Monaco, monospace;
            }
        """)
        splitter.addWidget(self.output_area)
        
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #222; }")
        
        splitter.setSizes([600, 200]) # orginal values 500 and 100
        
        #add a clear button
        clear_btn = QPushButton("Clear Output")
        clear_btn.clicked.connect(self.output_area.clear)
        
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(clear_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        #redirect stdout and stderr to the text area
        sys.stdout = OutputRedirector(self.output_area)
        sys.stderr = OutputRedirector(self.output_area)
        
    def closeEvent(self, event):
        if stream is not None:
            stream.stop()
            stream.close()
        event.accept()