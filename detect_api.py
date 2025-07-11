from flask import Flask, request, jsonify
import torch
from PIL import Image
from io import BytesIO
import pandas as pd  # Just to ensure pandas is included

app = Flask(__name__)  # ✅ Corrected

# ✅ Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    try:
        file = request.files['image']
        img = Image.open(BytesIO(file.read()))
        
        # Run detection
        results = model(img)
        
        # Parse results
        detections = results.pandas().xyxy[0].to_dict(orient='records')
        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':  # ✅ Corrected
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
