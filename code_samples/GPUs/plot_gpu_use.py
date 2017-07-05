import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


# parser
parser = argparse.ArgumentParser('Plot GPU use')
parser.add_argument('csv_path', type=str)
args = parser.parse_args()

with open(args.csv_path, 'r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  y = [[int(r) for r in row] for row in plots]
  x = range(len(y))  # number of iterations
  num_gpus = len(y[0])  # number of gpus

# plotting gpu use over time
plt.plot(x, y)
plt.legend(['gpu' + str(n) for n in range(num_gpus)])

plt.xlabel('time')
plt.ylabel('gpu use (%)')

plt.show()

# plotting average gpu use
avg_y = np.mean(np.asarray(y), axis=0)
y_pos = np.arange(num_gpus)
plt.bar(y_pos, height=avg_y)
plt.xticks(y_pos, ['gpu' + str(n) for n in range(num_gpus)])

plt.ylabel('Average gpu use (%)')

plt.show()
