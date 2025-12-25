from core.input_bus import bus, send, mark_assistant_ready #, debug_bus_id
from actions.actions import identifynetworkconnect
import psutil
from PySide6.QtCore import QObject, Signal, Slot, Property
import os 
import subprocess

def get_cpu_usage():
    """Non-blocking CPU usage. First call may return 0.0."""
    usage = psutil.cpu_percent(interval=None)
    return f"{usage:.1f}%"

class BackendBridge(QObject):
    messageReceived = Signal(str, str)  #role, text
    audioLevelChanged = Signal(float)
    cpuLoadChanged = Signal(str)
    networkStatusChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._bus = bus
        self._audio_level = 0.0
        self._cpu_load = "0%"
        self._network_online = False
        
        #debug_bus_id("BackendBridge")

        #Listen for backend output
        self._bus.assistant.connect(self._on_backend_output)

        #Tell input_bus that the GUI is ready
        mark_assistant_ready()
        
        #Prime the CPU meter (first call returns 0)
        psutil.cpu_percent(interval=None)

    #─────────────────────────────────────────────────────
    #Audio Level
    #─────────────────────────────────────────────────────
    @Property(float, notify=audioLevelChanged)
    def audioLevel(self):
        return self._audio_level

    @Slot(float)
    def setAudioLevel(self, level: float):
        """Called from listen.py to update audio level"""
        if self._audio_level != level:
            self._audio_level = level
            self.audioLevelChanged.emit(level)

    #─────────────────────────────────────────────────────
    #Chat
    #─────────────────────────────────────────────────────
    @Slot(str)
    def sendMessage(self, text: str) -> None:
        if not text.strip():
            return
        self.messageReceived.emit("me", text)
        send(text)

    @Slot(str)
    def _on_backend_output(self, text: str) -> None:
        """
        Called whenever the backend (print_to_gui / input_bus) emits text.

        We:
        - split multi-line blobs into separate messages
        - infer a role (user / assistant / system) from prefixes
        - emit each line to the QML model
        """
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue

            role = "assistant"

            #Heuristics for different prefixes that already exist in your logs
            if line.startswith("Me  -->") or line.startswith("YOU:"):
                role = "user"
                parts = line.split("-->", 1)
                if len(parts) == 2:
                    line = parts[1].strip()

            elif line.startswith("ARGUS:"):
                role = "assistant"
                line = line[len("ARGUS:"):].strip()

            elif (
                line.startswith("[HLRF]")
                or line.startswith("[DEBUG]")
                or line.startswith("[print_to_gui]")
            ):
                role = "system"

            #Everything still goes through to the GUI 
            self.messageReceived.emit(role, line)

    #─────────────────────────────────────────────────────
    #File System Operations
    #─────────────────────────────────────────────────────
    @Slot(str, result="QVariantList")
    def listDirectory(self, path: str):
        """List files/folders in a directory"""
        try:
            items = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                items.append({
                    "name": name,
                    "path": full,
                    "isDir": os.path.isdir(full)
                })
            return items
        except Exception as e:
            print(f"listDirectory error: {e}")
            return []

    @Slot(str)
    def openFile(self, path: str):
        """Open file with default app"""
        import sys
        if sys.platform == "darwin":
            subprocess.call(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.call(["xdg-open", path])

    @Slot(str, str, result="QVariantList")
    def searchFiles(self, root: str, query: str):
        """Search for files/folders"""
        matches = []
        q = query.lower()
        try:
            for root_dir, dirs, files in os.walk(root):
                for d in dirs:
                    if q in d.lower():
                        matches.append({"name": d, "path": os.path.join(root_dir, d), "isDir": True})
                for f in files:
                    if q in f.lower():
                        matches.append({"name": f, "path": os.path.join(root_dir, f), "isDir": False})
                if len(matches) >= 50:
                    break
        except:
            pass
        return matches[:50]

    @Slot(result=str)
    def getHomePath(self):
        return os.path.expanduser("~")
    
    #─────────────────────────────────────────────────────
    #System Stats
    #─────────────────────────────────────────────────────
    @Slot(result=str)
    def getCpuLoad(self) -> str:
        """Returns CPU load as a formatted string."""
        return get_cpu_usage()

    @Slot(result=str)
    def getNetworkStatus(self) -> str:
        """
        Returns network status as string.
        Compare with 'Connected' in QML for boolean checks.
        """
        status = identifynetworkconnect()
        #Normalize various possible return values
        if status and status.lower() in ("connected", "online", "true", "1"):
            return "Connected"
        return "Disconnected"

    @Slot(result=bool)
    def isNetworkConnected(self) -> bool:
        """Returns network status as boolean for easier QML binding."""
        status = identifynetworkconnect()
        if status and status.lower() in ("connected", "online", "true", "1"):
            return True
        return False