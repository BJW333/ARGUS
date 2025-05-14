import os
import math
import math
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QTimer, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QWidget


circle_base_radius = 100
circle_center = (300, 300)  #(x, y) not used directly here but you can adjust drawing logic
circles = []  #Will be generated in the custom widget
CHUNK = 1024
RATE = 44100
stream = None
is_thinking = False
is_listening = False
thinking_animation_step = 0

def audio_setup():
    global stream, CHUNK, RATE
    CHUNK = 1024  
    RATE = 44100  
    stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=CHUNK, dtype='int16')
    stream.start()

def update_audio_volume():
    global stream
    data, _ = stream.read(CHUNK)
    data = np.frombuffer(data, dtype=np.int16)
    volume = np.linalg.norm(data) / 10
    return volume

#GUI Components

class AnimatedCanvas(QWidget):
    def __init__(self, parent=None):
        super(AnimatedCanvas, self).__init__(parent)
        self.animation_step = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(20)  #update every 20ms

        #timer for audio driven update
        self.audio_timer = QTimer(self)
        self.audio_timer.timeout.connect(self.update)
        self.audio_timer.start(20)

    def update_animation(self):
        global is_thinking, thinking_animation_step
        if is_thinking:
            thinking_animation_step += 1
        else:
            thinking_animation_step = 0
        self.update()  #trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        #fill background with black
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        #get widget center
        center_x = self.width() // 2
        center_y = self.height() // 2

        #determine a radius based on audio volume
        volume = update_audio_volume() if stream is not None else 0
        #map volume to a radius value between 50 and 200
        radius = min(max(volume / 3000, 50), 200)

        #draw 5 circles with offsets and different colors
        for i in range(5):
            radius_offset = i * 10
            #if thinking apply a scale factor
            scale_factor = 1.05 + 0.05 * math.sin(thinking_animation_step / 10.0) if is_thinking else 1.0
            r = (circle_base_radius * scale_factor + radius_offset) + radius - 50

            #change outline color based on listening state
            if is_listening or volume > 300:
                r_color = max(0, 255 - i * 20)
                g_color = max(0, 255 - i * 30)
            else:
                r_color = max(0, 255 - i * 40)
                g_color = max(0, 255 - i * 50)
            color = QColor(r_color, g_color, 255)
            pen_width = max(1, 5 - i)
            pen = QPen(color, pen_width)
            painter.setPen(pen)

            #draw the ellipse centered in the widget
            rect = QRectF(center_x - r, center_y - r, 2 * r, 2 * r)
            painter.drawEllipse(rect)