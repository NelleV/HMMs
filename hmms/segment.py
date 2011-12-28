import numpy as np
from matplotlib import pyplot as plt


def get_patch(image):
    """
    """
    h, w = image.shape
    for i in range(w - 9):
        yield image[:, i:i + 9]


def split(image, thres=40, min_dist=10, max_dist=30):
    """
    """
    h, w = image.shape
    patches = get_patch(image[30:65, :])
    segments = []
    split_on = np.load('./hmms/data/split_on.npy')[30:65]

    for i, patch in enumerate(patches):
        score = ((patch - split_on) ** 2).sum()
        if score < thres or \
            (patch ** 2).sum() < thres:
            segments.append([score, i + 4])
    segments.sort()
    rsegments = []
    free = np.ones((w, 1))
    for _, element in segments:
        if free[element]:
            free[element - min_dist:element + min_dist] = 0
            rsegments.append(element)
    return rsegments


def show_segments(image, segments):
    """
    """
    image = image.copy()
    for segment in segments:
        image[:, segment] = 0
    plt.matshow(image)
