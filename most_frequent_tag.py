class MostFrequentTag(object):

    def __init__(self, train_data):
        self.train_data = train_data
        self.pos_tags_per_word = {}
        self.most_frequent_tag_per_word = {}
        self.pos_taggers_frequency = {}
        self.most_frequent_tag_overall = None

    def calculate_the_most_frequent_tag_per_word(self):

        for sentence in self.train_data:
            for word in sentence:
                if word[0] not in self.pos_tags_per_word:
                    self.pos_tags_per_word[word[0]] = {}
                if word[1] not in self.pos_tags_per_word[word[0]]:
                    self.pos_tags_per_word[word[0]][word[1]] = 1
                else:
                    self.pos_tags_per_word[word[0]][word[1]] += 1

        for word_tags in self.pos_tags_per_word:
            pos_tags_list = self.pos_tags_per_word.get(word_tags)
            key_max = max(pos_tags_list, key=lambda i: pos_tags_list[i])
            self.most_frequent_tag_per_word[word_tags] = key_max
            if key_max not in self.pos_taggers_frequency:
                self.pos_taggers_frequency[key_max] = pos_tags_list[key_max]
            else:
                self.pos_taggers_frequency[key_max] += pos_tags_list[key_max]

        self.most_frequent_tag_overall = max(self.pos_taggers_frequency, key=lambda i: self.pos_taggers_frequency[i])


    def predict(self, test_data):
        predicted_labels = []
        given_labels = []
        for sentence in test_data:
            current_labels = []
            current_labels_given = []
            for word in sentence:
                if word[0] in self.most_frequent_tag_per_word:
                    current_labels.append(self.most_frequent_tag_per_word[word[0]])
                else:
                    current_labels.append(self.most_frequent_tag_overall)
                current_labels_given.append(word[1])
            predicted_labels.append(current_labels)
            given_labels.append(current_labels_given)

        return [predicted_labels, given_labels]