import numpy as np
import os
import glob
import pickle

from scipy.misc import imresize
from sklearn.cluster import MiniBatchKMeans

from hmms.analyzer import LETTER_MAP
from hmms.utils import normalise

files = glob.glob('word_seg/done/*txt')
tmp_database = pickle.load(open('tmp_database.pck', 'r'))
database = pickle.load(open('database.pck', 'r'))
occurances = np.zeros((26, 5))

# we will have to resize
max_width = 22
max_height = 98

database = []
for element in files:
    labels = open(element, 'r').readline().split()
    num, filename = os.path.split(element)[-1].split('_')
    num = int(num)
    word = tmp_database[num][0]
    occ = False
    for label in labels:
        assert ((label == '_') or (label in word))
    for i, (label, im) in enumerate(zip(labels, tmp_database[num][1])):
        h, w = im.shape
        if i == 0:
            letter = [im]
        else:
            if label != previous_label:
                if previous_label == '_':
                    print "blank part"
                elif occ:
                    # we have a double letter here ! Hopefully, can split
                    # everything in two. Else...
                    occurances[LETTER_MAP[previous_label],
                               len(letter) / 2] += 1
                    occurances[LETTER_MAP[previous_label],
                               len(letter) - len(letter) / 2] += 1
                else:
                    occurances[LETTER_MAP[previous_label],
                               len(letter)] += 1
                letter = [im]
                word = word[1:]
                if len(word) > 1 and word[0] == word[1]:
                    occ = True
                else:
                    occ = False
            else:
                letter.append(im)
            if label != '_':
                database.append([label, im])
        previous_label = label
    if previous_label == '_':
        print "blank part"
    elif occ:
        # we have a double letter here ! Hopefully, can split
        # everything in two. Else...
        occurances[LETTER_MAP[previous_label],
                    len(letter) / 2] += 1
        occurances[LETTER_MAP[previous_label],
                    len(letter) - len(letter) / 2] += 1
    else:
        occurances[LETTER_MAP[previous_label],
                    len(letter)] += 1

pickle.dump(database, open('database.pck', 'w'))

# OK, let's create the descriptors array, to run kmeans.
# We'll have to resize a bunch of images, in order to have desc of all the
# same size.
desc = np.zeros((len(database), max_width * max_height))
for i, (_, element) in enumerate(database):
    el = imresize(element, (max_height, max_width))
    el = normalise(el)
    desc[i] = el.flatten()

k = min(200, len(desc))
km = MiniBatchKMeans(k=k)
km.fit(desc)

voc = km.cluster_centers_
labels = km.labels_
voc.dump('vocabulary.npy')

# OK. We have voc + labels. Let's compute the emission probability matrice
emission = np.zeros((26, len(voc)))
for el, label in zip(database, labels):
    emission[LETTER_MAP[el[0]], label] += 1

emission /= emission.sum(axis=1).reshape((len(emission), 1))
emission[np.isnan(emission)] = 0
emission.dump('emission.npy')

occurances /= occurances.sum(axis=1).reshape((len(occurances), 1))
occurances[np.isnan(occurances)] = 0
occurances.dump('occurances.npy')
