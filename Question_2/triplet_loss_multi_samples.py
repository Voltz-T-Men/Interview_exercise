import numpy as np

def triplet_loss_multi_samples(anchor, positives, negatives, margin=0.2):
    """
    Computes the Triplet Loss for multiple positives and negatives.

    Parameters:
    - anchor: numpy array of shape (m, n), embeddings for the anchor images
    - positive_distances: numpy array of shape (m, k, n), embeddings for the positive images
    - negative_distances:numpy array of shape (m, l, n), embeddings for the negative images
    - margin: alpha margin

    Returns:
    - loss: float, the value of the triplet loss.
    """
    # Tính L2 Distance giữa anchor và tất cả các positive embedding, rồi ta đem kết quả tính lấy trung bình
    positive_distances = np.mean(np.sum(np.square(anchor[:, np.newaxis, :] - positives), axis=2), axis=1)

    # Tính L2 Distance giữa anchor và tất cả các negative embedding, rrồi ta đem kết quả tính lấy trung bình
    negative_distances = np.mean(np.sum(np.square(anchor[:, np.newaxis, :] - negatives), axis=2), axis=1)

    # Tính toán Triplet Loss
    loss = np.maximum(0, positive_distance - negative_distance + margin)

    return np.mean(loss)