import numpy as np

def triplet_loss_one_sample(anchor, positive, negative, margin=1.0):
    """
    Computes the Triplet Loss for multiple anchors, positives, and negatives.

    Parameters:
    - anchor: np.ndarray, feature vector of the anchors.
    - positive_distance: np.ndarray, feature vector of the positive.
    - negative_distance: np.ndarray, feature vector of the negative.
    - margin: float, margin for calculating the loss.

    Returns:
    - total_loss: float, the value of the triplet loss.
    """
    # Tính L2 Distance giữa anchor và positive sample trong không gian embedding
    positive_distance = np.sum(np.square(anchor - positive), axis=-1)
    
    # Tính L2 Distance giữa anchor và negative sample trong không gian embedding
    negative_distance = np.sum(np.square(anchor - negative), axis=-1)
    
    # Tính toán Triplet Loss
    loss = np.maximum(0, positive_distance - negative_distance + margin)
    
    return loss