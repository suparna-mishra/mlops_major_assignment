import numpy as np

class QuantizedDecisionTree:
    def __init__(self, feature, threshold, children_left, children_right, value):
        self.feature = feature.astype(np.int16)
        self.threshold = threshold.astype(np.float32)
        self.children_left = children_left.astype(np.int16)
        self.children_right = children_right.astype(np.int16)
        self.value = value.astype(np.uint8)

    def predict_one(self, x):
        node = 0
        while self.children_left[node] != -1:
            feature = self.feature[node]
            th = self.threshold[node]
            if x[feature] <= th:
                node = self.children_left[node]
            else:
                node = self.children_right[node]
        return np.argmax(self.value[node])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])