# create metadata.tsv file
metadata_file = open('metadata.tsv', "w+")

# write header
metadata_file.write('Word\tIndex\n')

with open('data/vocab.txt', 'r') as vocab_file:
	index = 0
	for line in vocab_file:
		metadata_file.write(line[:len(line)-1] + '\t' + str(index) + '\n')
		index += 1

	
