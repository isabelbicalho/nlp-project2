# Computer Science Deparment - Universidade Federal de Minas Gerais
# 
# Natural Language Processing (2018/2)
# Professor: Adriano Veloso
#
# @author Isabel Amaro

import numpy as np

from utils import read_corpus
from utils import to_indexes
from utils import sentences_to_indexes
from utils import to_categorical
from utils import ignore_class_accuracy

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam


CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

print ("Loading datasets")
train_corpus, train_sentences, train_sentences_tags = read_corpus('../data/macmorpho-train.txt')
dev_corpus,   dev_sentences,   dev_sentences_tags = read_corpus('../data/macmorpho-train.txt')
test_corpus,  test_sentences,  test_sentences_tags = read_corpus('../data/macmorpho-train.txt')

print ("To indexes")
word2index, tag2index = to_indexes(train_sentences, train_sentences_tags)

train_sentences_X, train_tags_y = sentences_to_indexes(word2index, tag2index, train_sentences, train_sentences_tags)
dev_sentences_X, dev_tags_y = sentences_to_indexes(word2index, tag2index, dev_sentences, dev_sentences_tags)
test_sentences_X, test_tags_y = sentences_to_indexes(word2index, tag2index, test_sentences, test_sentences_tags)

MAX_LENGTH = len(max(train_sentences_X, key=len))

print ("MAX_LENGTH = {}".format(MAX_LENGTH))

print ("Padding")
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
dev_sentences_X  = pad_sequences(dev_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
dev_tags_y = pad_sequences(dev_tags_y, maxlen=MAX_LENGTH, padding='post')

print ("Building model")
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))

print ("Compiling model")
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])


#cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))

print ("Training model")
import pdb; pdb.set_trace()
model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=40, validation_split=0.2)

print ("Evaluating training")
scores = model.evaluate(train_sentences_X, to_categorical(train_tags_y, len(tag2index)))
print (scores)

#predictions = model.predict(dev_sentences_X)

print ("Predicting")
scores = model.evaluate(dev_sentences_X, to_categorical(dev_tags_y, len(tag2index)))
print (scores)
