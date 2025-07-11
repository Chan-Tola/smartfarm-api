from flask import Flask, request, jsonify
import torch
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load model - simple path since best.pt is in root folder
try:
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

@app.route('/detect', methods=['POST'])
def detect():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    try:
        file = request.files['image']
        img = Image.open(BytesIO(file.read()))

        # Run detection
        results = model(img)

        # Parse results
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        return jsonify({"detections": detections})
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)