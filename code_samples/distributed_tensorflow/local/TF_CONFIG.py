import json
import argparse

# parser
parser = argparse.ArgumentParser('Generate TF_CONFIG')

parser.add_argument('task_type', type=str, help='master, ps or worker')
parser.add_argument('task_index', type=int, help='should be a valid integer')

args = parser.parse_args()

# set up de the environment
cluster = {'master': ['localhost:2220'],
           'ps': ['localhost:2222', 'localhost:2221'],
           'worker': ['localhost:2223', 'localhost:2224']}

TF_CONFIG = json.dumps(
  {'cluster': cluster,
   'task': {'type': args.task_type, 'index': args.task_index},
   'model_dir': '/tmp/output_test',
   'environment': 'cloud'
  })

print(TF_CONFIG)
