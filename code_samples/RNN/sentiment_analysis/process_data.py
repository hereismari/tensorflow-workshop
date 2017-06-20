import argparse
import os

import re

# parser definition
parser = argparse.ArgumentParser(prog='process IMDB data')

parser.add_argument('data_path',
					type=str,
					help='Local path to the IMDB dataset in your machine')

parser.add_argument('--max_pad_len',
					type=int, default=80,
					help='Maximum length allowed for a setence, ' 
					      'if a len(sentence) > max_pad_len then '
					      'it will be padded. The default is 80')

parser.add_argument('--output_dir',
					type=str,
					help='Local path for the CSV outputs',
					default='data')

# constants
DATA_TYPE = ['train', 'test']
CLASS = ['neg', 'pos']
HEADER = ['review', 'is_positive', 'sequence_length']

def write_CSV_header(file_path):
	with open(file_path, "a+") as csv_file:
		csv_file.write(','.join(HEADER) + '\n')
	
# data_path: Local path to the IMDB dataset in your machine
# data_type: [train, test]
# cl: [neg, pos]
# output_dir: Output for the CSV files
# max_pad_len: default is 80
def process(data_path, data_type, cl, output_dir, max_pad_len, vocab_set):
	full_path = '{}/{}/{}/'.format(data_path, data_type, cl)
	output_path = '{}/{}.csv'.format(output_dir, data_type)
	
	# help keep track of processing
	number_of_files = len(os.listdir(full_path))
	files_processed = 0
	
	# save the vocabulary in a txt file
	VOCAB_PATH = '{}/{}.txt'.format(output_dir, 'vocab')
	vocab_file = open(VOCAB_PATH, "a+")
	
	print('Processing files at', full_path)
	with open(output_path, "a+") as csv_file:
		
		for filename in os.listdir(full_path):
			with open(full_path + filename) as f:
				
				sentence = f.readline()
				
				# remove <br> tags
				sentence = sentence.replace("<br>", " ")
				
				# using regex to remove any tipe of pontuation
				sentence = re.sub(r'[^a-zA-Z0-9 ]',r' ', sentence)
			
				# make all letters lower case
				sentence = sentence.lower()
				
				# use max_pad_len to make the sentence shorter 
				sentence = sentence.split()[:max_pad_len]
				
				# saving sequence length (number of words in the
				# sentence), because the Dynamic RNN uses it
				# to dynamiclly unroll the graph
				sequence_length = len(sentence)
				
				# update vocabulary
				for word in sentence:
					if word not in vocab_set:
						vocab_set[word] = 1
						vocab_file.write('{}\n'.format(word))
				
				sentence = ' '.join(sentence)
				
				# encode from 'pos' -> 1 and 'neg' -> 0
				is_positive = ('1' if cl == 'pos' else '0')
				
				csv_file.write('{},{},{}\n'.format(sentence,
												   is_positive,
												   sequence_length)) 
				
			files_processed += 1	
			if files_processed % 1000 == 0:
				print('Processed: %d/%d' % (files_processed, number_of_files))
				print('-' * 20)
			
def process_data_path(path):
	if path[-1] == '/': path = path[:len(path)-1]
	return path

def main():
	try:
		args = parser.parse_args()
	except:
		print(parser.print_help())
		exit()
		
	# get args
	data_path = process_data_path(args.data_path)
	output_dir = process_data_path(args.output_dir)
	max_pad_len = args.max_pad_len
		
	# dictionary to keep track of the vocabulary
	vocab_set = {}

	# write csv header for train and test files
	for dt in DATA_TYPE:
		write_CSV_header('{}/{}.csv'.format(output_dir, dt))

	# generate CSV files
	for dt in DATA_TYPE:
		for cl in CLASS:
			process(data_path, dt, cl, output_dir, max_pad_len, vocab_set)
	
if __name__ == '__main__':
	main()
