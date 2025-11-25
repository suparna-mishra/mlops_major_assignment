import joblib
from sklearn.metrics import accuracy_score

def test_model():
    model = joblib.load("savedmodel.pth")
    X_test, y_test = joblib.load("testset.pth")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("TEST ACCURACY:", acc)

if __name__ == "__main__":
    test_model()