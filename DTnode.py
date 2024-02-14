
class DTnode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value  # predicted class for Leaf Nodes
        self.left = left
        self.right = right
