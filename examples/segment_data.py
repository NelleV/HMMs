import numpy as np
import pickle
from hmms.data.load import load_letters, load_text_images
from hmms.segment import find_letters, split_on, find_words, show_segments
from hmms.analyzer import LETTER_MAP

database = []

def make_gen(bits):
    for bit in bits:
        yield bit

letters = load_letters()
letters_list = {}
for i in range(26):
    letters_list[i] = []

for i, letter in enumerate(letters):
    segments = find_letters(letter)
    bits = split_on(letter, segments)
    letters_list[i].append(bits)

# Computes the average space taken per letter
ave_letter = []
for letter in range(26):
    m = [j.shape[1] for l in letters_list[letter]
            for j in l]
    d = [len(l) for l in letters_list[letter]]
    ave_letter.append(
        [sum(m) / len(letters_list[letter]),
         sum(d) / len(letters_list[letter])])


# OK, now we have all the letters initialized
num = 0
texts = load_text_images()
for image, text in texts:
    words = text.split()
    segments = find_words(image)
    bits = split_on(image, segments, clean=True)
    # Just to check if we have segmented properly !
    show_segments(image, segments,
                  title=('./text_seg/%s' % text.replace(' ', '_')),
                  save=True)
    if len(bits) != len(words):
        print "problem with image %s" % text
        continue
    for im, word in zip(bits, words):
        h, w = im.shape
        seg = find_letters(im)
        el = split_on(im, seg)
        database.append([word, el])
        show_segments(im, seg,
                  title=('./word_seg/' + str(num) + '_' + word),
                  save=True)

        labels = []
        print ("computing word '%s' of length %d splitted in %d"
               " segments" %
               (word, len(word), len(el)))
        if len(word) == 1:
            # This is a one letter word - add it to the list. Else, we'll have
            # to do more complicated computation
            letters_list[LETTER_MAP[word]].append(el)
            labels.append(word)
        else:
            # We know for each letter how much space is taken approximatively.
            # Let's try to determine automatically what the segmentation did.
            av_word_length = sum([ave_letter[LETTER_MAP[i]][0] for i in word])
            av_word_split = sum([ave_letter[LETTER_MAP[i]][1] for i in word])

            print "image length", im.shape[1]
            print "Average word length %d, split %d" % (av_word_length,
                                                        av_word_split)
            gen = make_gen(el)
            disp = float((im.shape[1] - av_word_length)) / len(word)
            length = 0
            for l in word:
                length += ave_letter[LETTER_MAP[l]][0] + disp
                for b in gen:
                    if abs(length - b.shape[1]) < 2 or l == 'i':
                        labels.append(l)
                        length -= b.shape[1]
                        break
                    else:
                        labels.append(l)
                        length -= b.shape[1]
                    if length < 0:
                        break
        filename = open('./word_seg/'+ str(num) + '_' + word + '.txt', 'w')
        filename.write(' '.join(labels))
        num += 1


pickle.dump(database, open('tmp_database.pck', 'w'))
