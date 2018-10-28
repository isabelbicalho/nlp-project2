# Computer Science Deparment - Universidade Federal de Minas Gerais
# 
# Natural Language Processing (2018/2)
# Professor: Adriano Veloso
#
# @author Isabel Amaro

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
