import os
import io
import uuid
import h5py
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename

# Import utilities from model.py
from model import build_model, setup_model_config, load_classes, load_and_preprocess_image, predict_image

# Module-level cache for model and classes
_model = None
_classes = None
_IMAGE_SIZE = None

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Improved HTML page with modern, green-themed design
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Paddy Leaf Disease Detector 🌾</title>
  <style>
    body {
      font-family: 'Poppins', sans-serif;
      background: linear-gradient(135deg, #e3f9e5, #cdeac0);
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }
    h1 {
      color: #2e7d32;
      margin-top: 40px;
      font-size: 2.4rem;
      text-shadow: 1px 1px 2px #a5d6a7;
    }
    form {
      background: #ffffff;
      padding: 30px 40px;
      border-radius: 20px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.1);
      margin-top: 30px;
      text-align: center;
      transition: transform 0.2s ease;
    }
    form:hover {
      transform: scale(1.02);
    }
    input[type=file] {
      margin: 15px 0;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #66bb6a;
      background-color: #f9fff9;
      cursor: pointer;
    }
    input[type=submit] {
      background-color: #43a047;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 8px;
      cursor: pointer;
      transition: background-color 0.3s ease;
    }
    input[type=submit]:hover {
      background-color: #2e7d32;
    }
    hr {
      width: 80%;
      margin: 40px 0;
      border: 0;
      height: 1px;
      background: #81c784;
    }
    #result {
      text-align: center;
      margin-bottom: 50px;
      color: #1b5e20;
      max-width: 600px;
    }
    #result h2 {
      font-size: 1.8rem;
      color: #2e7d32;
      margin-bottom: 10px;
    }
    #result p {
      font-size: 1.2rem;
    }
    #result img {
      margin-top: 15px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
      max-width: 400px;
      width: 100%;
    }
    footer {
      margin-top: auto;
      padding: 15px;
      font-size: 0.9rem;
      color: #33691e;
    }
  </style>
</head>
<body>
  <h1>Paddy Leaf Disease Detector 🌿</h1>
  <form method="post" enctype="multipart/form-data" action="/predict">
    <input type="file" name="file" accept="image/*" required>
    <br>
    <input type="submit" value="Upload & Detect">
  </form>
  <hr>
  <div id="result">
    {% if result %}
      <h2>Prediction Result</h2>
      <p><strong>Disease:</strong> {{ result.label }}</p>
      <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
      <img src="/uploads/{{ filename }}" alt="Uploaded Rice Leaf">
    {% endif %}
  </div>
  <footer>© 2025 Paddy Leaf Disease Detector | Powered by CNN & TensorFlow 🌱</footer>
</body>
</html>
"""


def get_model():
    """Lazy-load the model and classes. Returns (model, classes, image_size)."""
    global _model, _classes, _IMAGE_SIZE
    if _model is None:
        model_path = 'rice_leaf_disease_model.h5'

        # Try to infer number of classes from the saved HDF5 weights file
        num_classes_from_h5 = None
        if os.path.exists(model_path):
            try:
                with h5py.File(model_path, 'r') as f:
                    if 'model_weights' in f:
                        layers = list(f['model_weights'].keys())
                        # search layers in reverse to find the final dense/kernel shape
                        for layer_name in reversed(layers):
                            layer_group = f['model_weights'][layer_name]
                            for ds_name in layer_group.keys():
                                if ds_name.endswith('kernel:0'):
                                    shape = layer_group[ds_name].shape
                                    if len(shape) == 2:
                                        num_classes_from_h5 = int(shape[1])
                                        break
                            if num_classes_from_h5 is not None:
                                break
            except Exception as e:
                print(f"Warning: could not inspect HDF5 model to infer num_classes: {e}")

        # Load classes.json if available (load_classes returns (cat_to_name, classes))
        try:
            cat_to_name, classes_list = load_classes('classes.json')
            _classes = classes_list
        except Exception:
            _classes = None

        # Determine number of classes to build the model with
        if num_classes_from_h5 is not None:
            num_classes = num_classes_from_h5
        elif _classes is not None:
            num_classes = len(_classes)
        else:
            raise RuntimeError('Unable to determine number of classes: provide classes.json or the HDF5 model file.')

        # If classes.json exists but does not match the model, override to match model
        if _classes is not None and len(_classes) != num_classes:
            print(f"Warning: 'classes.json' has {len(_classes)} classes but model expects {num_classes}. Overriding class names to match model.")
            _classes = [f"Class_{i}" for i in range(num_classes)]

        MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, BATCH_SIZE = setup_model_config()
        _IMAGE_SIZE = IMAGE_SIZE

        _model = build_model(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE, num_classes, trainable=False)
        if os.path.exists(model_path):
            try:
                _model.load_weights(model_path)
            except Exception as e:
                try:
                    print(f"Info: direct load_weights failed ({e}), trying by_name=True")
                    _model.load_weights(model_path, by_name=True)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load weights from {model_path}: {e2}")

    return _model, _classes, _IMAGE_SIZE


def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route('/')
    def index():
        return render_template_string(HTML_PAGE, result=None)

    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    @app.route('/predict', methods=['POST'])
    def predict_route():
        if 'file' not in request.files:
            return jsonify({'error': 'no file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'no selected file'}), 400

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        model, classes, image_size = get_model()
        img = load_and_preprocess_image(save_path, image_size)
        pred = predict_image(model, img, classes)
        label = list(pred.keys())[0]
        confidence = float(list(pred.values())[0]) * 100.0

        if 'text/html' in request.headers.get('Accept', ''):
            return render_template_string(
                HTML_PAGE,
                result={'label': label, 'confidence': f"{confidence:.2f}"},
                filename=unique_name
            )

        return jsonify({'label': label, 'confidence': confidence})

    return app


if __name__ == '__main__':
    app = create_app()
    print('Starting Flask server on http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=True)
