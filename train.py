import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from quantized_tree import QuantizedDecisionTree
import numpy as np

def train_model():
    data = fetch_olivetti_faces()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    tree = clf.tree_

    qtree = QuantizedDecisionTree(
        feature=tree.feature,
        threshold=tree.threshold,
        children_left=tree.children_left,
        children_right=tree.children_right,
        value=tree.value
    )

    joblib.dump(qtree, "savedmodel_quantized.pth")
    joblib.dump((X_test, y_test), "testset.pth")

    print("Quantized model saved")

if __name__ == "__main__":
    train_model()