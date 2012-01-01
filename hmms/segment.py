import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def get_patch(image, size=9):
    """
    """
    h, w = image.shape
    for i in range(w - size):
        yield image[:, i:i + size]


def find_letters(image, thres=35, min_dist=8, max_dist=30):
    """
    """
    h, w = image.shape
    patches = get_patch(image[30:65, :])
    blank_patches = get_patch(image[30:65], size=5)

    segments = []
    split_on = np.load('./hmms/data/split_on.npy')[30:65]
    split_on2 = np.load('./hmms/data/split_on_2.npy')[30:65]

    for i, patch in enumerate(blank_patches):
        bscore = ((1 - patch) ** 2).sum()
        score = ((split_on2 - patch) ** 2).sum()

        if score * 7. / 5 < thres or bscore < 1:
            segments.append([score, i + 2])

    for i, patch in enumerate(patches):
        score = ((patch - split_on) ** 2).sum()
        if score < thres:
            segments.append([score, i + 4])
    segments.sort()
    rsegments = []
    free = np.ones((w, 1))
    # We don't want something too close to the borders, so let's tag border
    # elements as not free
    free[:min_dist] = 0
    free[-min_dist:] = 0
    for _, element in segments:
        if free[element]:
            beg = max(0, element - min_dist)
            end = min(element + min_dist, w)
            free[beg:end] = 0
            rsegments.append(element)
    return rsegments


def find_words(image, thres=20, min_dist=15):
    """Find word in an image using whitespace seperation
    """
    # take "only" the middle, as often, j, and g go in the whitespace
    image = image[35:65, :]
    h, w = image.shape
    segments = []
    patches = get_patch(image, size=10)
    for i, patch in enumerate(patches):
        score = (1 - (patch ** 2)).sum()
        if score < thres:
            segments.append([score, i + 5])
    segments.sort()
    rsegments = []
    free = np.ones((w, 1))
    free[:min_dist] = 0
    free[-min_dist:] = 0
    for _, element in segments:
        if free[element]:
            free[element - min_dist:element + min_dist] = 0
            rsegments.append(element)
    return rsegments


def split_on(image, segments, clean=False):
    """
    Splits the image, and "clean" them, by trimming whitespace

    Parameters:
    -----------
        image
        segments

    Returns
    -------
    list of ndarray
    """
    images = []
    segments.sort()
    # FIXME clean up
    if segments:
        for i, segment in enumerate(segments):
            if i == 0:
                element = image[:, 0:segment]
            else:
                element = image[:, previous:segment]
            previous = segment

            if clean:
                element = clean_patch(element)

            images.append(element)
        element = image[:, previous:]
        if clean:
            element = clean_patch(element)
        images.append(element)
    else:
        element = clean_patch(image)
        images.append(element)
    return images


def clean_patch(element):
    # Let's start trimming:
    while element.shape[1] > 1 \
          and ((1 - element[:, 0]) ** 2).sum() < 5:
        element = element[:, 1:]

    while element.shape[1] > 1 \
          and ((1 - element[:, -1]) ** 2).sum() < 5:
        element = element[:, :-1]
    return element


def show_segments(image, segments, title="", save=False):
    """
    """
    for segment in segments:
        image[:, segment] = 0
    if not save:
        plt.matshow(image, cmap=cm.gray)
        plt.title(' '.join(title))
    else:
        plt.imsave(title + '.png', image, cmap=cm.gray)
