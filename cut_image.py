import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import numpy as np

# Cargar imagen
img_path = './images/00111_gt.png'  # Cambia esto si es necesario
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Variable para guardar coordenadas
coords = []

def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    coords.clear()
    coords.extend([x, y, w, h])
    print(f"Coordenadas seleccionadas: x, y, w, h = {x}, {y}, {w}, {h}")

# Mostrar imagen y activar selector
fig, ax = plt.subplots()
ax.imshow(img_rgb)
toggle_selector = RectangleSelector(ax, onselect,
                                     useblit=True,
                                     button=[1],  # click izquierdo
                                     minspanx=5, minspany=5,
                                     spancoords='pixels',
                                     interactive=True)
plt.show()
