import numpy as np
from scipy.misc import imresize

from sklearn.metrics import euclidean_distances

from hmms.segment import find_letters, split_on, show_segments, find_words
from hmms.data.load import load_text_images
from hmms import vdhmms
from hmms.utils import normalise
from hmms.analyzer import LETTER_MAP

print LETTER_MAP['a']

# Because we know it... FIXME
h, w = 98, 22
voc = np.load('vocabulary.npy')

text = load_text_images()
image, _ = text.next()
image, _ = text.next()

segments = find_words(image)

el = split_on(image, segments, clean=True)
im = el[0]

# load all the probability matrices we need
transition = np.load('transition.npy')
first_letter = np.load('first_letter.npy')
last_letter = np.load('last_letter.npy')
emission = np.load('emission.npy')
occurances = np.load('occurances.npy')
# Adding this by hand...
occurances[3, 0] += 0.5

emission[np.isnan(emission)] = 0

#im = imread('./data/and.png').mean(axis=2)
segments = find_letters(im)
patches = split_on(im, segments)

descs = np.zeros((len(patches), voc.shape[1]))
for i, patch in enumerate(patches):
    patch = imresize(patch, (h, w))
    patch = normalise(patch)
    descs[i] = patch.flatten()

# OK, these are the labels of our observation !
labels = euclidean_distances(descs, voc).argmin(axis=1)

# Now, we can compute the emission probability matrices
emission_obs = emission[:, labels].T

alphas = vdhmms.alpha(transition, emission_obs, occurances,
                      p_init=first_letter.reshape((len(first_letter), )))
betas = vdhmms.beta(transition, emission_obs, occurances,
                p_init=last_letter.reshape((len(first_letter), )))
show_segments(im, segments)

g = alphas * betas
chain = g.argmax(axis=1)
prob = g.max(axis=1)
# careful ! Some labels of the chain "just" correspond to null probabilities
letters = 'abcdefghijklmnopqrstuvwxyz'
labels = []
for p, l in zip(prob, chain):
    if p == 0:
       labels.append('_')
    else:
        labels.append(letters[l])


