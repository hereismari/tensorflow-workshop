#!/usr/bin/env python

import csv
import re

name_pattern = re.compile("^[a-zA-Z ]+$")
def name_valid(name):
	return re.match(name_pattern, name) is not None

def hex_to_rgb(hex):
    i = int(hex, 16)
    r = (i >> 16) & 0xFF
    g = (i >> 8) & 0xFF
    b = i & 0xFF
    return (r, g, b)


lineregex = re.compile("^([a-zA-Z ]+?)\s+?#([0-9a-f]{6})$")

infile = open("wikipedia.csv", "r")
outfile = open("db.csv", "w")
reader = csv.reader(infile)
csvwriter = csv.writer(outfile)

for line in reader:
    name = line[0]
    if not name_valid(name):
        continue
    name = name.lower()
    rgb = line[1].replace("#", "").lower()
    (r, g, b) = hex_to_rgb(rgb)

    csvwriter.writerow([name, r, g, b])
