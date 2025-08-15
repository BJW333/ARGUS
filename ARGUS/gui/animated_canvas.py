import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget
import pyqtgraph.opengl as gl

RATE = 44100 # Sample rate in Hz
CHUNK = 1024  # ~= 23.2 ms of audio; handled in callback thread
_stream = None
_vol_ema = 0.0   # exponential moving average of volume (thread-safe enough for reads)
_alpha = 0.25    # smoothing factor (0..1)
_ext_vol = None   # optional external volume (from SR), overrides when set

def _audio_callback(indata, frames, time, status):
    global _vol_ema
    if status:
        # drop status prints to avoid spam; you can log once if needed
        pass
    # RMS volume in 0..~1
    v = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
    # quick EMA to stabilize the pulse
    _vol_ema = _alpha * v + (1.0 - _alpha) * _vol_ema

def audio_setup():
    """Start a non-blocking input stream with a callback."""
    global _stream
    if _stream is None:
        try:
            _stream = sd.InputStream(
                samplerate=RATE,
                channels=1,
                blocksize=CHUNK,
                dtype='int16',
                callback=_audio_callback,
                latency='low',      # hint; helps on some devices
            )
            _stream.start()
        except Exception as e:
            # Don't spam prints; just fail open—UI will still run.
            _stream = None
            
def current_volume():
    """Read the latest smoothed volume without blocking the GUI thread."""
    # scale a bit to make the pulse visible; clamp to a sane range
    return max(0.0, min(1.0, _vol_ema * 12.0))

def mic_suspend():
    """Close the canvas mic so SR can claim the device (prevents AUHAL -50)."""
    global _stream
    if _stream:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
        _stream = None


def mic_resume():
    """Reopen the canvas mic after SR finishes."""
    audio_setup()


def mic_is_active() -> bool:
    return _stream is not None


def set_external_volume(v: float):
    """(Optional) Feed RMS from your SR path while the canvas mic is suspended."""
    global _ext_vol
    # clamp to 0..1
    _ext_vol = max(0.0, min(1.0, float(v)))
    
    
class AnimatedCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.gl_view = gl.GLViewWidget(self)
        self.gl_view.opts['distance'] = 9
        self.gl_view.setCameraPosition(elevation=25, azimuth=30)
        self.gl_view.setGeometry(self.rect())
        self.gl_view.setBackgroundColor((0, 0, 0, 0))

        audio_setup()

        self._angle = 0.0
        self._listening = False
        self._thinking  = False
        self._speaking  = False

        self.create_particles()
        self.create_rings()
        self.create_arcs()
        self.create_core_glow()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_scene)
        self.timer.start(16)  # ~60 FPS

    # Optional UI-state hooks
    def set_listening(self, on: bool): self._listening = bool(on)
    def set_thinking(self,  on: bool): self._thinking  = bool(on)
    def set_speaking(self,  on: bool): self._speaking  = bool(on)

    def create_particles(self):
        sizes = np.random.uniform(0.02, 0.06, 1600)
        pts = np.random.normal(size=(1600, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts *= np.random.uniform(0.8, 1.9, (1600, 1))
        self.scatter = gl.GLScatterPlotItem(
            pos=pts, size=sizes,
            color=(1.0, 0.85, 0.2, 0.8),
            pxMode=False
        )
        self.gl_view.addItem(self.scatter)

    def create_rings(self):
        self.rings = []
        self.ring_colors = []
        for i in range(14):
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.random.uniform(1.3, 2.3)
            ring = np.vstack([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]).T
            axis = np.random.rand(3); axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.pi)
            ring = ring @ self.rotation_matrix(angle, axis)
            color = (1.0, 0.78, 0.15, np.random.uniform(0.15, 0.4))
            item = gl.GLLinePlotItem(pos=ring, color=color, width=1.0, antialias=True)
            self.gl_view.addItem(item)
            self.rings.append(item)
            self.ring_colors.append(list(color))

    def create_arcs(self):
        self.arcs = []
        self.arc_colors = []
        for i in range(18):
            theta = np.linspace(0, np.pi, 50)
            r = np.random.uniform(1.0, 2.1)
            arc = np.vstack([r*np.sin(theta), np.zeros_like(theta), r*np.cos(theta)]).T
            axis = np.random.rand(3); axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2*np.pi)
            arc = arc @ self.rotation_matrix(angle, axis)
            color = (1.0, 0.7, 0.2, np.random.uniform(0.1, 0.3))
            item = gl.GLLinePlotItem(pos=arc, color=color, width=0.8, antialias=True)
            self.gl_view.addItem(item)
            self.arcs.append(item)
            self.arc_colors.append(list(color))

    def create_core_glow(self):
        self.core_glow = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=32, cols=32, radius=0.3),
            color=(1.0, 0.8, 0.2, 0.4),
            shader='balloon',
            smooth=True,
            glOptions='translucent'
        )
        self.gl_view.addItem(self.core_glow)

    def rotation_matrix(self, angle, axis):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        return np.array([
            [a*a+b*b-c*c-d*d, 2*(b*c-a*d),   2*(b*d+a*c)],
            [2*(b*c+a*d),     a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
            [2*(b*d-a*c),     2*(c*d+a*b),   a*a+d*d-b*b-c*c]
        ], dtype=float)

    def animate_scene(self):
        # non-blocking read of smoothed volume
        # If SR is providing volume (mic suspended), use that; otherwise use our own stream.
        global _ext_vol
        if _ext_vol is not None:
            vol = _ext_vol
            # decay external volume so the glow smoothly fades if SR stops feeding it
            _ext_vol *= 0.85
            if _ext_vol < 1e-3:
                _ext_vol = None
        else:
            vol = current_volume()

        # Emphasize mic pulse only when “listening”
        listen_boost = 0.35 if self._listening else 0.0
        speak_boost  = 0.20 if self._speaking  else 0.0

        scale = 1.0 + 0.40*np.sin(self._angle/15.0) + 0.30*vol + listen_boost + speak_boost
        scale = max(0.7, min(2.0, scale))

        self.core_glow.resetTransform()
        self.core_glow.scale(scale, scale, scale)

        for i, ring in enumerate(self.rings):
            ring.resetTransform()
            ring.rotate(self._angle*(0.3+0.02*i), 0, 1, 0)
            # update alpha only (avoid heavy geometry changes)
            a = 0.18 + 0.12*np.sin(self._angle/20.0 + i)
            col = self.ring_colors[i]; col[3] = a
            ring.setData(color=tuple(col))

        for i, arc in enumerate(self.arcs):
            arc.resetTransform()
            arc.rotate(self._angle*(0.25+0.01*i), 1, 0, 0)
            a = 0.1 + 0.12*np.sin(self._angle/25.0 + i)
            col = self.arc_colors[i]; col[3] = a
            arc.setData(color=tuple(col))

        self.scatter.resetTransform()
        self.scatter.rotate(self._angle*0.1, 0, 1, 0)

        self.gl_view.orbit(0.1, 0)
        self._angle += 0.5

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.gl_view.setGeometry(self.rect())

    def cleanup(self):
        self.timer.stop()
        global _stream
        if _stream:
            _stream.stop()
            _stream.close()
            _stream = None