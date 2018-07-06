import numpy as np
from nltk.tag import hmm
from nltk.tokenize import sent_tokenize
from nltk.probability import LidstoneProbDist

class HMMTagger(object):
    """ Base class for a POS tagger using Hidden Markov Models"""

    def __init__(self):
        """ Class Contructor """
        pass

    def train_tagger(self, train_data):
        """ Trains an hmm pos tagger"""
        self.trainer = hmm.HiddenMarkovModelTrainer()
        self.tagger = self.trainer.train_supervised(train_data)

    def tag_sentence(self, sentence):
        """ Returns tags for a single sentence"""
        if isinstance(sentence, list):
            return self.tagger.tag(sentence)
        else:
            return self.tagger.tag(sent_tokenize(sentence))
    
    def predict(self, test_data):
        """ Predict tags for test data """
        tags = []
        pred_tags = []
        for sentence in test_data:
            lemmas = [x[0] for x in sentence]
            tags.append([x[1] for x in sentence])
            pred_tags.append([pred[1] for pred in self.tag_sentence(lemmas)])

        return tags, pred_tags
