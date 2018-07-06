import sklearn_crfsuite


class CrfPosTagger(object):

    def __init__(self):
        pass

    def features(self, sentence, index):

        currWord = sentence[index][0]

        if (index > 0):
            prevWord = sentence[index - 1][0]
        else:
            prevWord = '<START>'

        if (index < len(sentence) - 1):
            nextWord = sentence[index + 1][0]
        else:
            nextWord = '<END>'

        return {
            'word': currWord,
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'curr_is_title': currWord.istitle(),
            'prev_is_title': prevWord.istitle(),
            'next_is_title': nextWord.istitle(),
            'curr_is_lower': currWord.islower(),
            'prev_is_lower': prevWord.islower(),
            'next_is_lower': nextWord.islower(),
            'curr_is_upper': currWord.isupper(),
            'prev_is_upper': prevWord.isupper(),
            'next_is_upper': nextWord.isupper(),
            'curr_is_digit': currWord.isdigit(),
            'prev_is_digit': prevWord.isdigit(),
            'next_is_digit': nextWord.isdigit(),
            'curr_prefix-1': currWord[0],
            'curr_prefix-2': currWord[:2],
            'curr_prefix-3': currWord[:3],
            'curr_suffix-1': currWord[-1],
            'curr_suffix-2': currWord[-2:],
            'curr_suffix-3': currWord[-3:],
            'prev_prefix-1': prevWord[0],
            'prev_prefix-2': prevWord[:2],
            'prev_prefix-3': prevWord[:3],
            'prev_suffix-1': prevWord[-1],
            'prev_suffix-2': prevWord[-2:],
            'prev_suffix-3': prevWord[-3:],
            'next_prefix-1': nextWord[0],
            'next_prefix-2': nextWord[:2],
            'next_prefix-3': nextWord[:3],
            'next_suffix-1': nextWord[-1],
            'next_suffix-2': nextWord[-2:],
            'next_suffix-3': nextWord[-3:],
            'prev_word': prevWord,
            'next_word': nextWord,
        }

    def transformDatasetSequence(self, sentences):
        wordFeatures, wordLabels = [], []

        for sent in sentences:
            feats, labels = [], []

            for index in range(len(sent)):
                feats.append(self.features(sent, index))
                labels.append(sent[index][1])

            wordFeatures.append(feats)
            wordLabels.append(labels)

        return wordFeatures, wordLabels

    def trainCRF(self, trainFeatures, trainLabels, crf_hyperparameters):
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=crf_hyperparameters[0],
            c2=crf_hyperparameters[1],
            max_iterations=100,
            all_possible_transitions=True,
        )
        crf.fit(trainFeatures, trainLabels)

        return crf