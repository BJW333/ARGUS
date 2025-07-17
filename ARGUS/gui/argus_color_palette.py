from PyQt5.QtGui import QColor

# ——— “Stark” HUD blue–cyan palette ————————————————————————————————
# Outline edges (panels, corner brackets, micro-text)
EDGE      = QColor(  0, 255, 255, 180)   # bright cyan
EDGE_DIM  = QColor(  0, 180, 255,  70)   # dimmer cyan for grids/slots

# Glass fills (very subtle cyanish tint)
FILL_1    = QColor(  5,  15,  25,  25)   # nearly‐transparent dark
FILL_2    = QColor( 20, 100, 255,  18)   # very pale blue

# Brain rings (wireframe)
BRAIN_RING= QColor(  0, 200, 255, 200)   # bright cyan–blue

# Neural connections glow (thinner, low alpha)
GLOW_LINE = QColor(  0, 200, 255,  60)   # pale cyan

# Drop shadow under brain/card
SHADOW    = QColor(  0,  40,  80, 160)   # dark bluish shadow