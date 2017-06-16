# isn't perfect, but gets mostly everything
import csv

import re

line_regex = re.compile('^"Resene ([a-zA-Z ]+?)"\s+?(\d{1,3})\s+?(\d{1,3})\s+(\d{1,3})$')

rawfile = open("resene.raw", "r")
outfile = open("db.csv", "w")
csvwriter = csv.writer(outfile)
for line in rawfile:
    line = line.strip()
    matches = re.match(line_regex, line)
    if matches is None:
        continue
    (name, r, g, b) = matches.groups()
    name = name.lower()

    # theres a few names with funky whitespacing, just kill off names that have more than three words
    if len(name.split(" ")) > 3:
        continue

    csvwriter.writerow([name, r, g, b])
