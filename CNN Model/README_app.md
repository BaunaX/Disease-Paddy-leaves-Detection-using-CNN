Rice Leaf Disease Predictor - Flask App

This small Flask app loads the trained model from `rice_leaf_disease_model.h5` and exposes a web UI to upload a rice leaf image and get a prediction.

Files added:
- `app.py` - Flask application. Import-safe: model is lazy-loaded on first request.
- `requirements.txt` - minimal Python dependencies.

Quick start
1. Create and activate a Python environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python "H:/CNN Model/app.py"
```

4. Open http://127.0.0.1:5000 in your browser, upload an image, and get a prediction.

Notes
- The app uses the `build_model` function from `model.py` and loads weights from `rice_leaf_disease_model.h5` into a fresh model instance. This avoids unsafe deserialization of Python lambdas in saved models.
- For production, consider converting the model to SavedModel format or serving via TF Serving.
