from flask import Flask, render_template, Response, jsonify
import cv2
from roboflow import Roboflow
import time

app = Flask(__name__)

# Biến toàn cục lưu số chỗ trống và chỗ đã đậu xe
parking_info = {
    "empty": 0,
    "occupied": 0
}

# Khởi tạo webcam và model Roboflow
cap = cv2.VideoCapture(0)
rf = Roboflow(api_key="dtVRBn5DVJVAYGcI3qJa")
workspace = rf.workspace()
project = workspace.project("parking-space-4-q3enj")
version = project.version(1)
model = version.model


def generate_frames():
    global parking_info
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Lưu tạm ảnh để gửi lên Roboflow
        cv2.imwrite("temp.jpg", frame)

        try:
            prediction = model.predict("temp.jpg", confidence=40, overlap=30).json()
        except Exception as e:
            print("Prediction error:", e)
            continue

        empty_count = 0
        occupied_count = 0

        # Vẽ bounding box và đếm số chỗ trống, đã có xe
        for obj in prediction.get("predictions", []):
            label = obj["class"]
            if label == "empty":
                empty_count += 1
            else:
                occupied_count += 1

            x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
            color = (0, 255, 0) if label == "empty" else (0, 0, 255)
            cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
            cv2.putText(frame, label, (x - w//2, y - h//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        parking_info["empty"] = empty_count
        parking_info["occupied"] = occupied_count

        # Encode frame sang jpeg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Trả về frame MJPEG để stream video
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/parking_info')
def get_parking_info():
    return jsonify(parking_info)


if __name__ == '__main__':
    app.run(debug=True)
