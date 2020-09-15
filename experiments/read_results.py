import numpy as np
import os
import argparse
from fnmatch import fnmatch

parser = argparse.ArgumentParser()
parser.add_argument('-p', help='Path to follow', default="")

args = parser.parse_args()

for path, subdirs, filenames in os.walk(os.path.join('results', args.p)):
    if filenames != []:
        scores = []
        for filename in filenames:
            if fnmatch(filename, 'seed*results.npy'):
                x = np.load(os.path.join(path, filename), allow_pickle=True
                            ).item()
                scores.append(x['score'])
        if len(scores) == 0:
            print('No valid results here: ', filenames)
        else:
            print('Score: ', np.mean(scores), 'Std: ', np.std(scores),
                  'Nb runs: ', len(scores), x['args'])
