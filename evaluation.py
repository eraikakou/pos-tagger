"""
	TODO: NA
    DONE:
        - Dynamic training sizes
"""
from crf_pos_tagger import CrfPosTagger
from hmm import HMMTagger
from sklearn_crfsuite import metrics
import matplotlib.pyplot as plt
from settings import SETTINGS


class Evaluation(object):

    def __init__(self):
        self.all_available_pos_tags_train = {}
        self.all_available_pos_tags_dev = {}

        self.train_sizes = SETTINGS['train_sizes']
        self.train_overall_accuracy = []
        self.dev_overall_accuracy = []
        self.train_f1_scores = {}
        self.train_precision = {}
        self.train_recall = {}
        self.dev_f1_scores = {}
        self.dev_precision = {}
        self.dev_recall = {}

    def classification_report_csv(self, report, train=True):
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = []
            row_data = line.split('      ')
            if train:
                if row_data[1] not in self.all_available_pos_tags_train:
                    self.all_available_pos_tags_train[row_data[1]] = []
                self.all_available_pos_tags_train[row_data[1]].append(row_data[3])
            else:
                if row_data[1] not in self.all_available_pos_tags_dev:
                    self.all_available_pos_tags_dev[row_data[1]] = []
                self.all_available_pos_tags_dev[row_data[1]].append(row_data[3])

    def plot_learning_curves(self, plot_title):
        plt.clf()
        plt.plot(self.train_sizes, self.train_overall_accuracy, label='Training overall accuracy')
        plt.plot(self.train_sizes, self.dev_overall_accuracy, label='Validation overall accuracy')

        plt.ylabel('Overall accuracy', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title(plot_title, fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, 1)
        plt.show()

    def computeSeqAccuracy(self, predictedTags, actualTags):
        total, correct = 0, 0

        for i in range(len(predictedTags)):
            for j in range(len(predictedTags[i])):
                total += 1
                if predictedTags[i][j] == actualTags[i][j]:
                    correct += 1

        return float(correct) / total

    def calculate_f1_stats(self, f1_score, train=True):
        """
            Get the precision, recall and f1 score values which are returned
            by the sklearn `flat_classfification_report` method and populate
            the relevant dictionaries

            The formet of the input string is the following:

                        precision    recall  f1-score   support

                    VBG       1.00      1.00      1.00        11
                    RB       1.00      1.00      1.00         9
                    -LRB-       1.00      1.00      1.00         1
                    MD       1.00      1.00      1.00         2
                    CD       1.00      1.00      1.00         4
                    ,       1.00      1.00      1.00         5
                    .       1.00      1.00      1.00        12
                    .....
                    .....
                    VBN       1.00      1.00      1.00         6
                    NNP       1.00      1.00      1.00        32
                    PRP       1.00      1.00      1.00         9

            avg / total       1.00      1.00      1.00       261

        """
        f1 = f1_score.split('\n')
        result = []
        for line in f1:
            words = []
            line = line.split(' ')
            for word in line:
                if word != '':
                    words.append(word)
            result.append(words)
        # Remove the first two and last two lines which we cannot use.
        result.pop(0)
        result.pop(0)
        result.pop()
        result.pop()
        result.pop()
        for res in result:
            if train:
                if res[0] in self.train_f1_scores:
                    self.train_precision[res[0]].append(res[1])
                    self.train_recall[res[0]].append(res[2])
                    self.train_f1_scores[res[0]].append(res[3])
                else:
                    self.train_precision[res[0]] = [res[1]]
                    self.train_recall[res[0]] = [res[2]]
                    self.train_f1_scores[res[0]] = [res[3]]

            else:
                if res[0] in self.dev_f1_scores:
                    self.dev_precision[res[0]].append(res[1])
                    self.dev_recall[res[0]].append(res[2])
                    self.dev_f1_scores[res[0]].append(res[3])
                else:
                    self.dev_precision[res[0]] = [res[1]]
                    self.dev_recall[res[0]] = [res[2]]
                    self.dev_f1_scores[res[0]] = [res[3]]

    def plot_f1_curves(self):
        for key in self.train_f1_scores.keys():
            plt.clf()
            plt.plot(self.train_sizes[-len(self.train_f1_scores[key]):], self.train_f1_scores[key], label='Training F1 score')
            plt.plot(self.train_sizes[-len(self.dev_f1_scores[key]):], self.dev_f1_scores[key], label='Validation F1 score')

            plt.ylabel('F1 Score', fontsize=14)
            plt.xlabel('Training set size', fontsize=14)
            title = 'F1 score for the: "' + str(key) + '" POS.'
            plt.title(title, fontsize=18, y=1.03)
            plt.legend()
            plt.ylim(0, 1)
            # plt.show()
            plt.savefig('plots/' + self.name + '_F1_score_POS_'+str(key)+'.png')



class HmmEvaluation(Evaluation):

    def __init__(self, train_data, dev_data):
        super().__init__()
        self.name = 'hmm'
        self.train_data = train_data
        self.dev_data = dev_data

    def calculate_overall_accuracy_and_f1_score_per_pos(self, print_results = False):
        hmm_tagger = HMMTagger()

        for i in self.train_sizes:
            hmm_tagger.train_tagger(self.train_data[:i])
            train_tags, train_pred = hmm_tagger.predict(self.train_data[:i])
            dev_tags, dev_pred = hmm_tagger.predict(self.dev_data)

            labels = []
            for sentence_tags in train_tags:
                for tag in sentence_tags:
                    labels.append(tag)
            labels = list(set(labels))

            accuracy_score_train = metrics.flat_accuracy_score(train_tags, train_pred)
            self.train_overall_accuracy.append(accuracy_score_train)
            accuracy_score_dev = metrics.flat_accuracy_score(dev_tags, dev_pred)
            self.dev_overall_accuracy.append(accuracy_score_dev)

            f1_score_train = metrics.flat_classification_report(train_tags, train_pred, labels = labels)
            self.calculate_f1_stats(f1_score_train)
            # self.classification_report_csv(f1_score_train)
            f1_score_dev = metrics.flat_classification_report(dev_tags, dev_pred, labels = labels)
            self.calculate_f1_stats(f1_score_dev, False)
            # self.classification_report_csv(f1_score_dev, False)

            if print_results:
                print('The overall accuracy on Train data for train size = ' + str(i) + ' is = ' + str(accuracy_score_train))
                print('The overall accuracy on DEV data for train size = ' + str(i) + ' is = ' + str(accuracy_score_dev))
                print('Report')
                print('The overall accuracy on Train data for train size = ' + str(i) + ' is = ' + f1_score_train)
                print('The overall accuracy on DEV data for train size = ' + str(i) + ' is = ' + f1_score_dev)
                print('--------------------------------------------------------------------------------------')


class CrfEvaluation(Evaluation):

    def __init__(self, trainSeqFeatures, trainSeqLabels, devSeqFeatures, devSeqLabels):
        super().__init__()
        self.name = 'crf'
        self.data_features = trainSeqFeatures
        self.data_target = trainSeqLabels
        self.dev_features = devSeqFeatures
        self.dev_target = devSeqLabels

    def calculate_overall_accuracy_and_f1_score_per_pos(self, crf_hyperparameters, print_results = False):
        crf_pos_model = CrfPosTagger()

        for i in self.train_sizes:
            my_model = crf_pos_model.trainCRF(self.data_features[:i], self.data_target[:i], crf_hyperparameters)
            train_pred = my_model.predict(self.data_features[:i])
            dev_pred = my_model.predict(self.dev_features)
            labels = list(my_model.classes_)

            accuracy_score_train = metrics.flat_accuracy_score(self.data_target[:i], train_pred)
            self.train_overall_accuracy.append(accuracy_score_train)
            accuracy_score_dev = metrics.flat_accuracy_score(self.dev_target, dev_pred)
            self.dev_overall_accuracy.append(accuracy_score_dev)

            f1_score_train = metrics.flat_classification_report(self.data_target[:i], train_pred, labels = labels)
            self.calculate_f1_stats(f1_score_train)
            # self.classification_report_csv(f1_score_train)
            f1_score_dev = metrics.flat_classification_report(self.dev_target, dev_pred, labels = labels)
            self.calculate_f1_stats(f1_score_dev, False)
            # self.classification_report_csv(f1_score_dev, False)

            if print_results:
                print('The overall accuracy on Train data for train size = ' + str(i) + ' is = ' + str(accuracy_score_train))
                print('The overall accuracy on DEV data for train size = ' + str(i) + ' is = ' + str(accuracy_score_dev))
                print('Report')
                print('The overall accuracy on Train data for train size = ' + str(i) + ' is = ' + f1_score_train)
                print('The overall accuracy on DEV data for train size = ' + str(i) + ' is = ' + f1_score_dev)
                print('--------------------------------------------------------------------------------------')

