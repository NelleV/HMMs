import numpy as np
from hmms.data.load import load_texts

LETTER_MAP = {'a': 0,
              'b': 1,
              'c': 2,
              'd': 3,
              'e': 4,
              'f': 5,
              'g': 6,
              'h': 7,
              'i': 8,
              'j': 9,
              'k': 10,
              'l': 11,
              'm': 12,
              'n': 13,
              'o': 14,
              'p': 15,
              'q': 16,
              'r': 17,
              's': 18,
              't': 19,
              'u': 20,
              'v': 21,
              'w': 22,
              'x': 23,
              'y': 24,
              'z': 25}

IGNORE = [',', ';', '.', "'", '-', '!', '?', ':']


def token_analyse():
    """
    Creates the probability matrices: transition, first_letter, last_letter

    returns
    -------
    transition, first_letter, last_letter
    """
    text = load_texts()

    transition = np.zeros((26, 26))
    first_letter = np.zeros((26, 1))
    last_letter = np.zeros((26, 1))
    for line in text:
        line = line.lower()
        for element in IGNORE:
            line = line.replace(element, ' ')
        words = line.split()
        for word in words:
            for i, letter in enumerate(word):
                if i == 0:
                    first_letter[LETTER_MAP[letter]] += 1
                else:
                    transition[previous, LETTER_MAP[letter]] += 1
                previous = LETTER_MAP[letter]

            last_letter[LETTER_MAP[letter]] += 1
    # Now, let's create probabilities:
    transition /= transition.sum(axis=1).reshape((26, 1))
    # Some letters may not appear
    transition[np.isnan(transition)] = 0
    first_letter /= first_letter.sum()
    last_letter /= last_letter.sum()
    return transition, first_letter, last_letter
