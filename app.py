from flask import Flask, request
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model (Nano version is smallest - you can try yolov8s.pt, etc.)
model = YOLO('yolov8n.pt')

@app.route('/', methods=['GET'])
def home_page():
    # Simple HTML page with an upload form
    return '''
<html>
    <head>
        <title>YOLOv8 Food Detector</title>
    </head>
    <body>
        <h1>Welcome to the YOLOv8 Food Detector</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <p>Select an image file to upload:</p>
            <input type="file" name="file" />
            <button type="submit">Upload</button>
        </form>
    </body>
</html>
'''

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file part is present
    if 'file' not in request.files:
        return '''
<html>
    <body>
        <p>No file part in the request</p>
        <p><a href="/">Go Back</a></p>
    </body>
</html>
'''

    file = request.files['file']
    image_bytes = file.read()

    # Convert bytes to a CV2 image (BGR format)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference with a confidence threshold
    results = model.predict(source=img, conf=0.5)
    detections = results[0]

    detected_foods = []
    for box in detections.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[class_id]
        if conf > 0.5:
            detected_foods.append(class_name)

    # Convert the uploaded image to base64 for display
    base64_img = base64.b64encode(image_bytes).decode('utf-8')
    data_url = 'data:image/jpeg;base64,' + base64_img

    # Build a results page that shows detected items and the uploaded image
    return '''
<html>
    <head>
        <title>Detection Results</title>
    </head>
    <body>
        <h1>Detected foods: ''' + ', '.join(detected_foods) + '''</h1>
        <p><img src="''' + data_url + '''" alt="Uploaded Image" /></p>
        <p><a href="/">Go Back</a></p>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
