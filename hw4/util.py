#!/usr/bin/env python
__author__ = 'Yasheed'
# File script.py


if __name__ == '__main__':
	buffer_ = []
	texts = []
	texts_labels = {}
	with open('data/training_label.txt', 'r') as f:
		buffer_ = f.read()
		print(len(buffer_.split('\n')))
		for line in buffer_.split('\n'):
			if line == "":
				break
			index_of_comma = line.find(',')
			texts.append(line[index_of_comma:])
	print len(texts)