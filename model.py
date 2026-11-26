"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if len(np.unique(labels)) == 1 or len(labels) == 0:
        return 0.0
    l, counts = np.unique(labels, return_counts=True)
    p = counts / len(labels)

    H = -float(
        np.sum(p * np.log(p))
        )

    return H

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    mask = features[:, feature_index] > threshold # list of true false indicies
    lx = features[~mask]
    ly = labels[~mask]
    rx = features[mask]
    ry = labels[mask]
    return lx, ly, rx, ry

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)
    w_l = len(left_labels) / n
    w_r = len(right_labels) / n 
    return impurity(parent_labels) - (w_l * impurity(left_labels) + w_r * impurity(right_labels))

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    current_best = {
        'feature_index': None, 
        'threshold': None, 
        'score': 0.0}

    for fi in feature_indices:
        u = np.unique(features[:, fi])
        if len(u) <= 1:
            continue
        # Try midpoints between consecutive unique values
        for i in range(len(u) - 1):
            thresh = (u[i] + u[i + 1]) / 2.0
            lx, ly, rx, ry = split_dataset(features, labels, fi, thresh)
            if len(lx) == 0 or len(rx) == 0:
                continue
            score = split_score(labels, ly, ry)
            if score > current_best['score']:
                current_best['feature_index'] = fi
                current_best['threshold'] = thresh
                current_best['score'] = score

    return current_best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    if len(np.unique(labels)) == 1 or depth >= max_depth or len(labels) < min_samples_split:
        return True
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    labels = np.array(labels, dtype=int)
    unique, count = np.unique(labels, return_counts=True)
    indices = np.where(count == np.max(count))[0]
    return int(unique[indices[0]])

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes. 
    feature_ind = list(range(0, features.shape[-1])) if feature_subset is None else list(feature_subset)
    bst = best_split(features, labels, feature_ind)
    if should_stop(labels, depth, max_depth, min_samples_split) or bst['score'] <= 0.0:
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
            }

    lx, ly, rx, ry = split_dataset(features, labels, bst['feature_index'], bst['threshold'])
    l = build_tree(lx, ly, max_depth, min_samples_split, feature_subset, depth+1)
    r = build_tree(rx, ry, max_depth, min_samples_split, feature_subset, depth+1)

    return {
        'leaf': False,
        'feature_index': bst['feature_index'],
        'threshold': bst['threshold'],
        'left': l,
        'right': r
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    while not tree['leaf']: 
        j = tree['feature_index']
        t = tree['threshold']
        v = example[j]

        if v <= t:
            tree = tree['left']
        else:
            tree = tree['right']
    return int(tree['prediction'])

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    preds = []
    for i in features:
        preds.append(predict_example_tree(tree, i))
    return np.array(preds, dtype=int)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    n = features.shape[0]

    indices = rng.integers(0, n, size=n)
    sampled_features = features[indices]
    sampled_labels = labels[indices]

    return (sampled_features, sampled_labels)

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

