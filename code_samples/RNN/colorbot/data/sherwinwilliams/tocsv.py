#!/usr/bin/env python

import re
import csv

# isn't perfect, but gets mostly everything
line_regex = re.compile('^([\d]{4})\s+?([a-zA-Z ]+?)\s+?(\d{1,3})\s+?(\d{1,3})\s+(\d{1,3})$')

rawfile = open("paints.raw", "r")
outfile = open("db.csv", "w")
csvwriter = csv.writer(outfile)

colors_written = {}
for line in rawfile:
	line = line.strip()
	matches = re.match(line_regex, line)
	if matches is None:
		continue
	(code, name, r, g, b) = matches.groups()
	name = name.lower()
	
	# theres a few names with funky whitespacing, just kill off names that have more than three words
	if len(name.split(" ")) > 3:
		continue

	if name not in colors_written:
		colors_written[name] = 1
		csvwriter.writerow([name, r, g, b])
