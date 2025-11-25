import joblib
from sklearn.metrics import accuracy_score
import os

def test_model():
    model = joblib.load("savedmodel_quantized.pth")
    X_test, y_test = joblib.load("testset.pth")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Quantized Accuracy:", acc)
    import os

def size(path):
    s = os.path.getsize(path)
    return f"{s/1024:.2f} KB"

print("Original model:", size("savedmodel.pth"))
print("Quantized model:", size("savedmodel_quantized.pth"))

if __name__ == "__main__":
    test_model()