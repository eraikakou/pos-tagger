""" Simulation Entrypoint """

import sklearn_crfsuite
import random
from data_loader import DataLoader
from most_frequent_tag import MostFrequentTag
from crf_pos_tagger import CrfPosTagger
from evaluation import HmmEvaluation, CrfEvaluation, Evaluation
from hmm import HMMTagger
from sklearn_crfsuite import metrics


def main():
	print("Starting app...")

	SEED = 56427
	random.seed(SEED)

	data = DataLoader()
	data.load_data()

	# Create baseline model predict, print accuracy
	print('\n Baseline Model')
	print('---------------------------------------------------')
	most_freq_tag_baseline = MostFrequentTag(data.train_data)
	most_freq_tag_baseline.calculate_the_most_frequent_tag_per_word()
	predicted_labels, given_labels = most_freq_tag_baseline.predict(data.test_data)
	eval = Evaluation()
	baseline_accuracy = eval.computeSeqAccuracy(predicted_labels, given_labels)
	print('The accuracy of Most Frequent Tag baseline is: ' + str(round(baseline_accuracy, 2)) + '%')

	baseline_labels = []
	for sentence_tags in given_labels:
		for tag in sentence_tags:
			if tag not in baseline_labels:
				baseline_labels.append(tag)

	baseline_f1_score_train = metrics.flat_classification_report(given_labels, predicted_labels, labels=baseline_labels)
	print(baseline_f1_score_train)

	# Create Hmm model, train, predict, print accuracy
	hmm_tagger = HMMTagger()
	hmm_tagger.train_tagger(data.train_data)
	dev_tags, dev_pred = hmm_tagger.predict(data.dev_data)

	hmmeval = HmmEvaluation(data.train_data, data.dev_data)
	hmmeval.calculate_overall_accuracy_and_f1_score_per_pos()
	hmmeval.plot_learning_curves('The learning curves for hmm pos tagger model')
	hmmeval.plot_f1_curves()
	hmm_accuracy = hmmeval.computeSeqAccuracy(dev_pred, dev_tags)
	print('The accuracy of hmm Pos Tagger is: ' + str(round(hmm_accuracy, 2) * 100) + '%')

	# Create crf model, train, predict, print accuracy
	print('\n Crf Pos Tagger')
	print('---------------------------------------------------')
	print('Create the features for crf pos tagger model for train, dev and test set.\n')
	crf_pos = CrfPosTagger()
	trainSeqFeatures, trainSeqLabels = crf_pos.transformDatasetSequence(data.train_data)
	devSeqFeatures, devSeqLabels = crf_pos.transformDatasetSequence(data.dev_data)
	testSeqFeatures, testSeqLabels = crf_pos.transformDatasetSequence(data.test_data)

	print('Tune the hyper-parameters c1 and c2 of crf model on a held-out part of the training data, the dev set.\n')
	crf_hyperparameters = tune_the_hyperparameters_on_held_out_data(trainSeqFeatures, trainSeqLabels, devSeqFeatures, devSeqLabels)
	print('The best value for c1 is ' + str(crf_hyperparameters[0]) + ' anc c2 = ' + str(crf_hyperparameters[1]) + '. \n')
	print('Train crf using the best hyperparameters. \n')
	crf_model = crf_pos.trainCRF(trainSeqFeatures, trainSeqLabels, crf_hyperparameters)

	crfeval = CrfEvaluation(trainSeqFeatures, trainSeqLabels, devSeqFeatures, devSeqLabels)
	crfeval.calculate_overall_accuracy_and_f1_score_per_pos(crf_hyperparameters)
	crfeval.plot_learning_curves('The learning curves for crf pos tagger model')
	crfeval.plot_f1_curves()

	print('Predict the labels on test set. \n')
	pred_labels = crf_model.predict(testSeqFeatures)
	crf_accuracy = crfeval.computeSeqAccuracy(pred_labels, testSeqLabels)
	print('The accuracy of crf Pos Tagger is: ' + str(round(crf_accuracy, 2) * 100) + '%')
	labels = list(crf_model.classes_)

	f1_score_train = metrics.flat_classification_report(testSeqLabels, pred_labels, labels=labels)
	print(f1_score_train)

def tune_the_hyperparameters_on_held_out_data(trainFeatures, trainLabels, devSeqFeatures, devSeqLabels):
	"""
	Tune the hyper-parameters on a held-out part of the training data
	"""
	scores = {}
	param_grid = {
		'c1': [0.1, 1, 0.01],
		'c2': [0.1, 1, 0.01]
	}

	for c1 in param_grid['c1']:
		for c2 in param_grid['c2']:
			crf = sklearn_crfsuite.CRF(
				algorithm='lbfgs',
				c1=c1,
				c2=c2,
				max_iterations=100,
				all_possible_transitions=True,
			)

			crf.fit(trainFeatures, trainLabels)

			pred_labels = crf.predict(devSeqFeatures)
			current_parameters = [c1, c2]
			crfeval = CrfEvaluation(trainFeatures, trainLabels, devSeqFeatures, devSeqLabels)
			held_out_score = crfeval.computeSeqAccuracy(pred_labels, devSeqLabels)
			if held_out_score not in scores:
				scores[held_out_score] = current_parameters

	max_acurracy = max(scores)

	return scores[max_acurracy]


if __name__ == '__main__':
	main()