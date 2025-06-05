import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget
import pyqtgraph.opengl as gl

CHUNK, RATE = 1024, 44100
stream = None

def audio_setup():
    global stream
    if stream is None:
        stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=CHUNK, dtype='int16')
        stream.start()

def update_audio_volume():
    global stream
    if stream:
        data, _ = stream.read(CHUNK)
        samples = np.frombuffer(data, dtype=np.int16)
        return np.linalg.norm(samples) / 3000.0
    return 0.0

class AnimatedCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.gl_view = gl.GLViewWidget(self)
        self.gl_view.opts['distance'] = 9 #orginally was set at 12
        self.gl_view.setCameraPosition(elevation=25, azimuth=30)
        self.gl_view.setGeometry(self.rect())
        self.gl_view.setBackgroundColor((0, 0, 0, 0))

        audio_setup()

        self._angle = 0.0

        self._create_particles()
        self._create_rings()
        self._create_arcs()
        self._create_core_glow()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_scene)
        self.timer.start(16)

    def _create_particles(self):
        sizes = np.random.uniform(0.02, 0.06, 1600)
        pts = np.random.normal(size=(1600, 3))
        pts /= np.linalg.norm(pts, axis=1).reshape(-1, 1)
        pts *= np.random.uniform(0.8, 1.9, (1600, 1))
        self.scatter = gl.GLScatterPlotItem(
            pos=pts, size=sizes,
            color=(1.0, 0.85, 0.2, 0.8),
            pxMode=False
        )
        self.gl_view.addItem(self.scatter)

    def _create_rings(self):
        self.rings = []
        self.ring_colors = []
        for _ in range(14):
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.random.uniform(1.3, 2.3)
            ring = np.vstack([r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta)]).T
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.pi)
            ring = np.dot(ring, self._rotation_matrix(angle, axis))
            color = (1.0, 0.78, 0.15, np.random.uniform(0.15, 0.4))
            item = gl.GLLinePlotItem(pos=ring, color=color, width=1.0, antialias=True)
            self.gl_view.addItem(item)
            self.rings.append(item)
            self.ring_colors.append(color)

    def _create_arcs(self):
        self.arcs = []
        self.arc_colors = []
        for _ in range(18):
            theta = np.linspace(0, np.pi, 50)
            r = np.random.uniform(1.0, 2.1)
            arc = np.vstack([r*np.sin(theta), np.zeros_like(theta), r*np.cos(theta)]).T
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2*np.pi)
            arc = np.dot(arc, self._rotation_matrix(angle, axis))
            color = (1.0, 0.7, 0.2, np.random.uniform(0.1, 0.3))
            item = gl.GLLinePlotItem(pos=arc, color=color, width=0.8, antialias=True)
            self.gl_view.addItem(item)
            self.arcs.append(item)
            self.arc_colors.append(color)

    def _create_core_glow(self):
        self.core_glow = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=32, cols=32, radius=0.3),
            color=(1.0, 0.8, 0.2, 0.4),  # GOLD not GREEN
            shader='balloon',
            smooth=True,
            glOptions='translucent'
        )
        self.gl_view.addItem(self.core_glow)

    def _rotation_matrix(self, angle, axis):
        axis /= np.linalg.norm(axis)
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        return np.array([
            [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
            [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
            [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
        ])

    def _animate_scene(self):
        volume = update_audio_volume()
        scale_factor = 1.0 + 0.4*np.sin(self._angle/15) + 0.3*volume

        # Core pulse
        self.core_glow.resetTransform()
        self.core_glow.scale(scale_factor, scale_factor, scale_factor)

        # Rotate and subtle opacity modulation for rings
        for i, ring in enumerate(self.rings):
            ring.resetTransform()
            ring.rotate(self._angle*(0.3+0.02*i), 0, 1, 0)
            modulated_alpha = 0.2 + 0.15*np.sin(self._angle/20 + i)
            ring.setData(color=(1.0, 0.78, 0.15, modulated_alpha))

        # Rotate and opacity for arcs
        for i, arc in enumerate(self.arcs):
            arc.resetTransform()
            arc.rotate(self._angle*(0.25+0.01*i), 1, 0, 0)
            modulated_alpha = 0.1 + 0.15*np.sin(self._angle/25 + i)
            arc.setData(color=(1.0, 0.7, 0.2, modulated_alpha))

        # Particle rotation
        self.scatter.resetTransform()
        self.scatter.rotate(self._angle*0.1, 0, 1, 0)

        self.gl_view.orbit(0.1, 0)
        self._angle += 0.5

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.gl_view.setGeometry(self.rect())

    def cleanup(self):
        self.timer.stop()
        global stream
        if stream:
            stream.stop()
            stream.close()
            stream = None