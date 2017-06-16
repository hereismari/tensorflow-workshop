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


fp = open("all.js", "r")
raw = json.load(fp)
raw = raw["colors"][1:]  # trim header
fp.close()

colors_written = {}

outfile = open("db.csv", "w")
csvwriter = csv.writer(outfile)
for color in raw:
    hex = color[16][1:]  # strip hashtag
    name = color[19].lower().strip()  # name
    if not name_valid(name):
        continue

    if name not in colors_written:
        colors_written[name] = 1
        (r, g, b) = hex_to_rgb(hex)
        csvwriter.writerow([name, r, g, b])
