import torch
from torch import nn
import torch.functional as F
import numpy as np
import pandas as pd


def get_nums(config):
    path = config.movie_lens_path
    data = pd.read_csv(path, sep = ',')#[:1000] #slice for testing
    return (len(pd.unique(data["item"])), len(pd.unique(data["user"])))


def random_neq(l, r, s):
    '''function from original code to get negative samples
    '''
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def truncate_pad(seq, maxlen):
    '''truncates or pads sequences to give the correct sequence length'''

    if len(seq) > maxlen:
        seq = seq[-maxlen:]
    if len(seq) < maxlen:
        pad = [0 for i in range(maxlen - len(seq))]
        seq = pad + seq
    return seq