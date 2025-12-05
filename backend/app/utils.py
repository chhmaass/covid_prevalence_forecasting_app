# Utility functions placeholder
def normalize_weights(w1, w2):
    total = w1 + w2
    if total <= 0:
        return 0.5, 0.5
    return w1 / total, w2 / total
