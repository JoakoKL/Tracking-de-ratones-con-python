from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Video source
VIDEO_PATH = "NOR-1-571.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Background subtractor (MOG2)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Lista para guardar el historial de puntos (x, y)
track_points = []

import time
...
def gen_video():
    """Stream del vídeo original con caja y centro dibujado."""
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Máscara de movimiento
        fgmask = fgbg.apply(frame)
        # Limpieza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Encontrar contornos
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Escoger el contorno más grande (asumimos que es el ratón)
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            track_points.append((cx, cy))

            # Dibujar
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

        # Encode frame
        ret2, jpeg = cv2.imencode('.jpg', frame)

        # Ajustar velocidad de reproducción según los FPS del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        time.sleep(1.0 / fps)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def gen_track():
    """Stream de la trayectoria en un lienzo separado."""
    canvas_size = (480, 640, 3)
    while True:
        # lienzo blanco
        canvas = 255 * np.ones(canvas_size, dtype=np.uint8)

        # Dibujar línea de trayectoria
        if len(track_points) > 1:
            for i in range(1, len(track_points)):
                cv2.line(canvas,
                         track_points[i-1],
                         track_points[i],
                         (255,0,0), 2)

        # Punto actual
        if track_points:
            cv2.circle(canvas, track_points[-1], 5, (0,0,255), -1)

        ret, jpeg = cv2.imencode('.jpg', canvas)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Página con dos pestañas (video + trayectoria)."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/track_feed')
def track_feed():
    return Response(gen_track(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

...
@app.route('/reset', methods=['POST'])
def reset():
    """Reinicia el video a cero y limpia el lienzo"""
    global cap, track_points
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    track_points.clear()
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    """Ruta para recibir el video subido y actualizar la fuente."""
    global cap, VIDEO_PATH
    if 'video' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Guarda el archivo en la carpeta actual o en una ruta deseada
    filename = file.filename
    file.save(filename)
    
    # Actualiza la fuente del video
    VIDEO_PATH = filename
    cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)

    track_points.clear()
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)