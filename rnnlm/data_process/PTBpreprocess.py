# encoding=utf-8

import os
import string

source_filename = 'ptb.test.txt'
output_filename = 'preprocessed_ptb_data.tsv'
max_sentence_len = 79
# max_sentence_len = 0

f1 = open(source_filename, 'r')
all_data = f1.read()
all_data = all_data.split('\n')
f1.close()

word_to_index = {}
index_to_word = {}
word_counter = 0

word_to_index['<BOS>'] = word_counter
index_to_word[word_counter] = '<BOS>'
word_counter += 1
word_to_index['<EOS>'] = word_counter
index_to_word[word_counter] = '<EOS>'
word_counter += 1
word_to_index['<PAD>'] = word_counter
index_to_word[word_counter] = '<PAD>'
word_counter += 1
for sentence in all_data:
	# if max_sentence_len < len(sentence.strip().split(' ')):
	# 	max_sentence_len = len(sentence.strip().split(' '))
	for word in sentence.strip().split(' '):
		if not word in word_to_index.keys() and len(word) > 0:
			word_to_index[word] = word_counter
			index_to_word[word_counter] = word
			word_counter += 1

# print max_sentence_len
print word_counter
# for item in sorted(word_to_index.keys()):
# 	print item

f2 = open(output_filename, 'w')
for line in all_data:
	f2.write(str(word_to_index['<BOS>']) + ' ')
	len_counter = 1
	for word in line.strip().split(' '):
		if len(word) > 0:
			f2.write(str(word_to_index[word]) + ' ')
			len_counter += 1
	f2.write(str(word_to_index['<EOS>']) + ' ')
	len_counter += 1
	while len_counter < max_sentence_len:
		f2.write(str(word_to_index['<PAD>']) + ' ')
		len_counter += 1
	f2.write('\n')
f2.close()

