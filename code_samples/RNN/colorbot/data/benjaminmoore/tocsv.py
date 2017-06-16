#!/usr/bin/env python

import json
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

colors = {}
with open("./colors.json", "r") as colorfile:
    colors = json.load(colorfile)
    colors = colors["colors"]

colors_written = {}
with open("db.csv", "wb") as csvfile:
	writer = csv.writer(csvfile)
	for code in colors:
		color = colors[code]
		name = color["name"]
		
		if not name_valid(name):
			continue
		
		name = name.lower()

		if name not in colors_written:
			colors_written[name] = 1
			hex = color["hex"]
			(r, g, b) = hex_to_rgb(hex)
			writer.writerow([name, r, g, b])
