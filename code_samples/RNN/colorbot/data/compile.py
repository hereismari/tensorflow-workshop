#!/usr/bin/env python
# original code from: https://github.com/andrewortman/colorbot
# modified by @mari-linhares

TEST_SIZE=100

import glob
import random
import csv

def write_csv(name, colors):
  fp = open(name, "w")
  writer = csv.writer(fp)
  writer.writerow(["name", "red", "green", "blue"])
  for color in colors:
    writer.writerow(color)
  fp.close()

def write_vocab(name, vocab):
  keys = vocab.keys()
  keys.sort()
  fp = open(name, "wb")
  fp.write("\n".join(keys))
  fp.close()

allcolors = []
vocab = {}
color_names = {}
lines_seen = {}

duplicates = 0
for file in glob.glob("./*/db.csv"):
  fp = open(file, "rb")
  reader = csv.reader(fp)
  for line in reader:
    color = line[0]

    red = line[1]
    green = line[2]
    blue = line[3]

    for char in color:
      if char not in vocab:
        vocab[char] = 1

    if color in color_names:
      print "ignoring duplicate: " + color + ", " + str(color_names[color])
      continue

    color_names[color] = 1

    line_merged = ";".join(line)
    lines_seen[line_merged] = 1

    allcolors.append([color, red, green, blue])
  fp.close()

random.shuffle(allcolors)

write_csv("test.csv", allcolors[0:TEST_SIZE])
write_csv("train.csv", allcolors[TEST_SIZE:])
write_vocab("vocab.txt", vocab)
