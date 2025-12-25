import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from core.bootstrap import initialize_argus_backend
from gui_qml.backend_bridge import BackendBridge

def main():
    #initialize ARGUS backend services
    initialize_argus_backend()

    #set high DPI scaling policy
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    #create the application instance
    app = QGuiApplication(sys.argv)
    app.setApplicationName("ARGUS")
    
    #create QML engine and set up backend bridge
    engine = QQmlApplicationEngine()

    backend = BackendBridge()
    backend.setObjectName("BackendBridge")
    app.backend = backend
    engine.rootContext().setContextProperty("Backend", backend)

    qml_file = Path(__file__).parent / "gui_qml" / "MainView.qml"
    engine.load(str(qml_file))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()