import random
import numpy as np

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
			if value > 5:
				self.working_vocabulary[key] = counter
				self.working_vocabulary_inverse.append(key)
				counter += 1

		self.paraphrase_blocks_length = len(self.paraphrase_blocks);

	#word to one hot vector length of the vocab
	def word_to_index(self, word):
		# 1 extra for unk, one extra for *PAD*
		if word == "*PAD*":
			return len(self.working_vocabulary) + 1

		if word in self.working_vocabulary:
			return self.working_vocabulary[word]
		else:
			return len(self.working_vocabulary)

	# back from one hot to word
	def index_to_word(self, index):
		if index == len(self.working_vocabulary) + 1:
			return "*PAD*"
		if index == len(self.working_vocabulary):
			return "*UNK*"
		return self.working_vocabulary_inverse[index]

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
 

	def one_hot_to_word(self, one_hot):
 		ind = one_hot.index(1)
 		if ind == len(self.working_vocabulary) + 1:
 			return "*PAD*"
 		if ind == len(self.working_vocabulary):
 			return "*UNK*"
 		return self.working_vocabulary_inverse[ind]
 
 	#string to list of onehot vectors size of working vocab, length is 15 with padding
 	def sentence_to_one_hot(self, sentence):
 		arr = sentence.split()
 		one_hots = list(map(lambda w : self.word_to_one_hot(w), arr))
 		if len(one_hots) < 15:
 			one_hots = one_hots + [self.word_to_one_hot("*PAD*")]*(15-len(one_hots))
 		return one_hots

	#string to list of indices in working vocab, length is 15 with padding
	def sentence_to_indices(self, sentence):
		arr = sentence.split()
		indices = map(lambda w : self.word_to_index(w), arr)
		if len(indices) < 15:
			indices = indices + [self.word_to_index("*PAD*")]*(15-len(indices))
		return indices

	def get_random_positive_batch(self, batch_size):
		batch1 = []
		batch2 = []
		for i in range(batch_size):
			block = random.sample(self.paraphrase_blocks, 1)
			samp = random.sample(block[0], 2)
			batch1.append(self.sentence_to_one_hot(samp[0]))
			batch2.append(self.sentence_to_one_hot(samp[1]))
		return [np.array(batch1),np.array(batch2)]

	def get_random_negative_batch(self, batch_size):
		batch1 = []
		batch2 = []
		for i in range(batch_size):
			blocks = random.sample(self.paraphrase_blocks,2)
			samp1 = random.sample(blocks[0],1)
			samp2 = random.sample(blocks[1],1)
			batch1.append(self.sentence_to_one_hot(samp1[0]))
			batch2.append(self.sentence_to_one_hot(samp2[0]))
		return [np.array(batch1),np.array(batch2)]


if __name__ == '__main__':
	p = ProcessData()
	p.process("train_set.txt")

	print (len(p.working_vocabulary))
	pos = p.get_random_positive_batch(1)
	pos1 = map(lambda oh : p.one_hot_to_word(list(oh)), pos[0][0])
	pos2 = map(lambda oh : p.one_hot_to_word(list(oh)), pos[1][0])

	print(pos1)
	print(pos2)

	pos = p.get_random_negative_batch(1)
	pos1 = map(lambda oh : p.one_hot_to_word(list(oh)), pos[0][0])
	pos2 = map(lambda oh : p.one_hot_to_word(list(oh)), pos[1][0])

	print(pos1)
	print(pos2)