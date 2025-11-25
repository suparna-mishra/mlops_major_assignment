from flask import Flask, request, render_template
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

model = joblib.load("savedmodel.pth")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("L").resize((64, 64))
    arr = np.array(img).reshape(1, -1)

    pred = model.predict(arr)[0]
    return f"Predicted Class: {pred}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)