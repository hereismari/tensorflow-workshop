#!/usr/bin/env python

import csv
import re

def hex_to_rgb(hex):
        i = int(hex, 16)
        r = (i >> 16) & 0xFF
        g = (i >> 8) & 0xFF
        b = i & 0xFF
        return (r, g, b)


lineregex = re.compile("^([a-zA-Z ]+?)\s+?#([0-9a-f]{6})$")

infile = open("rgb.txt", "r")
outfile = open("db.csv", "w")
csvwriter = csv.writer(outfile)

for line in infile:
	line = line.strip()
	matches = re.match(lineregex, line)
	if matches is None:
		print line
		continue
		
	(name, hex) = matches.groups()

	name = name.lower()
	(r, g, b) = hex_to_rgb(hex)

	csvwriter.writerow([name, r, g, b])
