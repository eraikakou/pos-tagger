"""
	Settings file for our exercise
	------------------------------
	* input_files_dir
		The directory which contains the data for our exercise. They are taken from
		https://github.com/UniversalDependencies/UD_English-EWT
	* train_file, dev_file, test_file
		The names of the corresponding data sets.
	* fields
		The fields we will take into account for our processing. ORDERING MATTERS!
		In case you want to increase the fields, just append the list. This requires
		though further functionality on the application itself. It has been implemented
		this way for ease of expandability.
	* train_sizes
		The list holding the various number of sentences to use for training.

"""

SETTINGS = {
	'input_files_dir': '../UD_English-EWT/',
	'train_file': 'en-ud-train.conllu',
	'dev_file': 'en-ud-dev.conllu',
	'test_file': 'en-ud-test.conllu',
	'fields': ['lemma', 'xpostag'],
	'train_sizes': [1, 100, 500, 1000, 2000, 5000, 6000, 8000]
}