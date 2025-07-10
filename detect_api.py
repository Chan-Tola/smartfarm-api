from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load model (updated path to root directory)
model = YOLO('best.pt')

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
        detections = results[0].pandas().xyxy[0].to_dict(orient="records")
        return jsonify({"detections": detections})
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
