import math
import numpy as np
import os, shutil
import sys

# taken from https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
def print_progress_bar(i,end):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*(int)(i*20/(end-1)), 5*(20/(end-1))*i))
    sys.stdout.flush()

# taken from https://stackoverflow.com/questions/50637446/computing-euclidean-distance-with-multiple-list-in-python
def distance_lists(list1, list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5

def clear_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
