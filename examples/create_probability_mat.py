import numpy as np

from hmms.analyzer import token_analyse

transition, first_letter, last_letter = token_analyse()

transition.dump('transition.npy')
first_letter.dump('first_letter.npy')
last_letter.dump('last_letter.npy')
