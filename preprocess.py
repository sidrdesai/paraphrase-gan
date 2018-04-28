import random

class ProcessData:
	def __init__(self):
		#frequencies of each word in the corpus
		self.vocabulary_counts = {}
		#vocab excluding words that only appear once
		self.working_vocabulary = {}
		self.working_vocabulary_inverse = []
		#list of list of strings
		self.paraphrase_blocks = []

		self.paraphrase_blocks_length = 0

	def process(self, file):
		paraphrase_block = []
		with open(file) as f:
			for line in f:
				if line == "\n":
					if len(paraphrase_block) == 1:
						paraphrase_block = []
					else:
						self.paraphrase_blocks.append(paraphrase_block)
						paraphrase_block = []
				else:
					paraphrase_block.append(line)
					arr = line.split()
					for word in arr:
						if word in self.vocabulary_counts:
							self.vocabulary_counts[word] += 1
						else:
							self.vocabulary_counts[word] = 1
		counter = 0
		for key, value in self.vocabulary_counts.items():
			if value > 1:
				self.working_vocabulary[key] = counter
				self.working_vocabulary_inverse.append(key)
				counter += 1

		self.paraphrase_blocks_length = len(self.paraphrase_blocks);

	#word to one hot vector length of the vocab
	def word_to_one_hot(self, word):
		# 1 extra for unk, one extra for *PAD*
		to_return = [0]*(len(self.working_vocabulary) + 2)
		if word == "*PAD*":
			to_return[len(to_return) - 1] = 1
			return to_return

		if word in self.working_vocabulary:
			to_return[self.working_vocabulary[word]] = 1
			return to_return
		else:
			to_return[len(to_return) - 2] = 1
			return to_return

	# back from one hot to word
	def one_hot_to_word(self, one_hot):
		ind = one_hot.index(1)
		if ind == len(self.working_vocabulary) + 1:
			return "*PAD*"
		if ind == len(self.working_vocabulary):
			return "*UNK*"
		return self.working_vocabulary_inverse[ind]

	#string to list of onehot vectors size of working vocab, length is 15 with padding
	def sentence_to_onehot(self, sentence):
		arr = sentence.split()
		one_hots = list(map(lambda w : self.word_to_one_hot(w), arr))
		if len(one_hots) < 15:
			one_hots = one_hots + [self.word_to_one_hot("*PAD*")]*(15-len(one_hots))
		return one_hots

	def get_random_positive_batch(self, batch_size):
		batch = []
		for i in range(batch_size):
			block = random.sample(self.paraphrase_blocks, 1)
			samp = random.sample(block[0], 2)
			batch.append((self.sentence_to_onehot(samp[0]),self.sentence_to_onehot(samp[1])))
		return batch

	def get_random_negative_batch(self, batch_size):
		batch = []
		for i in range(batch_size):
			blocks = random.sample(self.paraphrase_blocks,2)
			samp1 = random.sample(blocks[0],1)
			samp2 = random.sample(blocks[1],1)
			batch.append((self.sentence_to_onehot(samp1[0]),self.sentence_to_onehot(samp2[0])))
		return batch


if __name__ == '__main__':
	p = ProcessData()
	p.process("train_set.txt")


	pos = p.get_random_positive_batch(1)[0]
	pos1 = list(map(lambda oh : p.one_hot_to_word(oh), pos[0]))
	pos2 = list(map(lambda oh : p.one_hot_to_word(oh), pos[1]))

	print(pos1)
	print(pos2)

	pos = p.get_random_negative_batch(1)[0]
	pos1 = list(map(lambda oh : p.one_hot_to_word(oh), pos[0]))
	pos2 = list(map(lambda oh : p.one_hot_to_word(oh), pos[1]))

	print(pos1)
	print(pos2)