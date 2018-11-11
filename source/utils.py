# Computer Science Deparment - Universidade Federal de Minas Gerais
# 
# Natural Language Processing (2018/2)
# Professor: Adriano Veloso
#
# @author Isabel Amaro

import numpy as np
from keras import backend as K


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def read_corpus(filename):
    with open(filename, "r") as corpus:
        lines = corpus.read().splitlines()
    sentences = [line.split() for line in lines]
    corpus = [list(map(lambda x: tuple(x.split("_")), sentence)) for sentence in sentences]
    sentences, sentences_tags = [], []
    for tagged_sentence in corpus:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentences_tags.append(np.array(tags))
    return corpus, sentences, sentences_tags


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    import pdb; pdb.set_trace()
    return np.array(cat_sequences)


def to_indexes(sentences, sentences_tags):
    words, tags = set([]), set([])
    for sentence in sentences:
        for word in sentence:
            words.add(word.lower())
    for sentence_tag in sentences_tags:
        for tag in sentence_tag:
            tags.add(tag)
    word2index = {word: i + 2 for i, word in enumerate(list(words))}
    word2index['-PAD-'] = 0 # Padding
    word2index['-OOV-'] = 1 # Out of vocabulary
    tag2index = {tag: i + 1 for i, tag in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # Padding
    return word2index, tag2index


def sentences_to_indexes(word2index, tag2index, sentences, sentences_tags):
    sentences_X, tags_y = [], []
    for sentence in sentences:
        sentence_int = []
        for word in sentence:
            try:
                sentence_int.append(word2index[word.lower()])
            except KeyError:
                sentence_int.append(word2index['-OOV-'])
        sentences_X.append(sentence_int)

    for sentence_tags in sentences_tags:
        tags_y.append([tag2index[tag] for tag in sentence_tags])

    return sentences_X, tags_y


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
        token_sequences.append(token_sequence)
    return token_sequences


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy
