from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

modelo = YOLO("yolov8n.pt")

camara = cv2.VideoCapture(0)

def generar_frames():
    while True:
        success, frame = camara.read()

        if not success:
            break

        resultados = modelo(frame)

        frame = resultados[0].plot()

        cv2.putText(
            frame,
            "CONTROL DE ACCESO INDUSTRIAL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(
        generar_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
