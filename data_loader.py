
from settings import SETTINGS
from conllu import parse, parse_tree

class DataLoader(object):
	""" """

	def __init__(self):
		self.files_dir  = SETTINGS['input_files_dir']
		self.train_file = self.files_dir + SETTINGS['train_file']
		self.dev_file   = self.files_dir + SETTINGS['dev_file']
		self.test_file  = self.files_dir + SETTINGS['test_file']

		print("Files of simulation:")
		print(self.train_file)
		print(self.dev_file)
		print(self.test_file)

	def load_data(self):
		with open(self.train_file,"r", encoding="utf8") as f:
			self.train_data = f.read()

		with open(self.dev_file,"r", encoding="utf8") as f:
			self.dev_data = f.read()

		with open(self.test_file,"r", encoding="utf8") as f:
			self.test_data = f.read()

		# Parse the data
		self.train_data = parse(self.train_data)
		self.dev_data   = parse(self.dev_data)
		self.test_data  = parse(self.test_data)

		sentences_train = [sentence for sentence in self.train_data]
		sentences_dev   = [sentence for sentence in self.dev_data]
		sentences_test  = [sentence for sentence in self.test_data]
		fields    = [field for field in SETTINGS['fields']]

		# the main lists
		self.train_data = []
		self.dev_data   = []
		self.test_data  = []

		for sentence in sentences_train:
			# the sentence list
			cur_sent = []
			for word in sentence:
				# the word list
				cur_set = [word[field] for field in fields]
				# append the word list to the current sentence list
				cur_sent.append(cur_set)
			# append the sentence list to the main one
			self.train_data.append(cur_sent)

		for sentence in sentences_dev:
			cur_sent = []
			for word in sentence:
				cur_set = [word[field] for field in fields]
				cur_sent.append(cur_set)
			self.dev_data.append(cur_sent)

		for sentence in sentences_test:
			cur_sent = []
			for word in sentence:
				cur_set = [word[field] for field in fields]
				cur_sent.append(cur_set)
			self.test_data.append(cur_sent)

