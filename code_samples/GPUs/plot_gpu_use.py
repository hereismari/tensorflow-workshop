import matplotlib.pyplot as plt
import csv
import argparse

# parser
parser = argparse.ArgumentParser('Plot GPU use')
parser.add_argument('csv_path', type=str)
args = parser.parse_args()

with open(args.csv_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    
    y = [[r for r in row] for row in plots]
    x = range(len(y)) # number of iterations
    num_gpus = len(y[0]) # number of gpus

plt.plot(x, y)
plt.legend(['gpu' + str(n) for n in range(num_gpus)])

plt.xlabel('time')
plt.ylabel('gpu use (%)')

plt.show()
