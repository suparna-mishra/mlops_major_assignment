try:
    import joblib
except ImportError:
    try:
        from sklearn.externals import joblib
    except ImportError:
        raise ImportError("joblib is required; install it with 'pip install joblib'")

try:
    from sklearn.datasets import fetch_olivetti_faces
except ImportError:
    raise ImportError("scikit-learn is required; install it with 'pip install scikit-learn'")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_model():
    data = fetch_olivetti_faces()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    joblib.dump(clf, "savedmodel.pth")
    joblib.dump((X_test, y_test), "testset.pth")

    print("Model trained and saved as savedmodel.pth")

if __name__ == "__main__":
    train_model()