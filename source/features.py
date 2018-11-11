# Computer Science Deparment - Universidade Federal de Minas Gerais
# 
# Natural Language Processing (2018/2)
# Professor: Adriano Veloso
#
# @author Isabel Amaro

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing      import LabelEncoder
from keras.utils                import np_utils

class FeatureEngineer(object):

    @staticmethod
    def add_basic_features(sentence_terms, index):
        term = sentence_terms[index]
            return {
                'nb_terms': len(sentence_terms),
                'term': term,
                'is_first': index == 0,
                'is_last': index == len(sentence_terms) - 1,
                'is_capitalized': term[0].upper() == term[0],
                'is_all_caps': term.upper() == term,
                'is_all_lower': term.lower() == term,
                'prefix-1': term[0],
                'prefix-2': term[:2],
                'prefix-3': term[:3],
                'suffix-1': term[-1],
                'suffix-2': term[-2:],
                'suffix-3': term[-3:],
                'prev_word': '' if index == 0 else sentence_terms[index - 1],
                'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
            }

    @staticmethod
    def untag(tagged_sentence):
        return [w for w, _ in tagged_sentence]

    @staticmethod
    def transform_to_dataset(tagged_sentences):
        X, y = [], []

        for pos_tags in tagged_sentences:
            for index, (term, class_) in enumerate(pos_tags):
                # Add basic NLP features for each sentence term
                X.append(add_basic_features(untag(pos_tags), index))
                y.append(class_)
        return X, y

    @staticmethod
    def encode(X, y):
        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(X)
        X = dict_vectorizer.transform(X)
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        y = label_encoder.transform(y)
        y = np_utils.to_categorical(y)
        return X, y
