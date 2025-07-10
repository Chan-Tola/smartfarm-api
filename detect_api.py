from flask import Flask, request, jsonify
import torch
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/smartfarm_model/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    img = Image.open(BytesIO(file.read()))

    # Run detection
    results = model(img)

    # Parse results
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)