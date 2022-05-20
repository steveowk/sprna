from itertools import islice
from glob import glob
from csv import writer
import random
import math 
import torch
import torch.nn as nn
from collections import deque
from torch.nn.utils.rnn import pad_sequence


single_sites = ["G","A","U","C"]
paired_sites = ["GC","CG","AU","UA","UG","GU"]
single_sites_ix = {ix:x for ix, x in enumerate(single_sites)}
paired_sites_ix = {ix:x for ix, x in enumerate(paired_sites)}
all_sites_ix = {ix:x for ix, x in enumerate(single_sites+paired_sites)}

EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0


def sampleReplay(positive, negative, sz):
    sample_size = sz
    if len(positive) >= sample_size:
        neg_sample_train = random.sample(negative, sample_size)
        pos_sample_train = random.sample(positive, sample_size)
    elif len(positive) == 0: # if there is no positive yet
        sample_size = 10
        neg_sample_train = random.sample(negative, sample_size)
        pos_sample_train = []
    else:
        sample_size = len(positive)
        neg_sample_train = random.sample(negative, sample_size)
        pos_sample_train = random.sample(positive, sample_size)
    
    X , y = [], []
    for d in pos_sample_train+neg_sample_train:
        X.append(d[0])
        y.append(d[1])        
    X = pad_sequence(X, padding_value=-1)
    X = X.view(X.size(1),1,X.size(0),X.size(2))
    y = torch.cat(y).view(-1, 1)
    assert(X.size(0)==y.size(0))
    return X,y

def evalArgmax(preds):
    out = nn.Softmax(dim=0)(preds)
    return torch.argmax(out).item()

def decayingEgreedy(preds):
    global steps_done
    eps_threshold = EPS_END+(EPS_START-EPS_END)*math.exp(-1.*steps_done/EPS_DECAY)
    steps_done+=1
    if random.random() > eps_threshold:
        with torch.no_grad():
            out = nn.Softmax(dim=0)(preds)
            return torch.argmax(out).item()
    return random.choice(list(range(len(preds))))

def trackEvalResults():
    track_results = {
                "datasetC": set(),
                "datasetA": set(),
                "datasetB": set(),
                "datasetD": set()
                }
    return track_results

def loadTrain():
    return loadCandD("train")

def loadValidation():
    eval_data = {"datasetC": loadCandD("modena"),
                 "datasetA": loadAandB("antarnav"),
                 "datasetB": loadAandB("antarnat"),
                 "datasetD": loadCandD("test")
              }
    assert(len(eval_data['datasetD'])==100 and len(eval_data['datasetB'])==83 and len(eval_data['datasetC'])==29)
    return eval_data

def loadCandD(data):
    path = f"data/{data}/*.rna"
    all_files =  glob(path)
    targets = []
    for _ , r in enumerate(all_files):
        with open(r, "r") as myfile:
            lines = myfile.readlines()
            if not lines:continue
            assert(len(lines)==1)
            lines = lines[0].strip()
            targets.append(lines)
    return targets

def loadAandB(data):
    if data == "antarnav" : path = "./data/antarnav.txt"
    elif data == "antarnat" :path = "./data/antarnat.txt"
    targets = []
    with open(path, "r") as myfile:
        for line in myfile.readlines():
            line = line.strip()
            if line.startswith('.') or line.startswith('('):
                targets.append(line)
    return targets

def binaryCodings(seq):
    def to_bin(x):return '{0:04b}'.format(x)
    binary_codes = dict()
    for ix, x in enumerate(seq) : binary_codes[seq[ix]] = to_bin(ix)
    binary_codes['unknown_pair'] = to_bin(10)
    binary_codes['unknown_single'] = to_bin(11)
    return binary_codes

def generateW(seq, n):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def getAllPairings(target):
    stack = []
    paired_bases= list()
    unpaired_bases = list()
    for i in range(len(target)):
        if target[i] == '(':
            stack.append(i)
        if target[i] == ')':
            paired_bases.append((stack.pop(), i))
        elif target[i]=='.':
            unpaired_bases.append(i)
    del stack
    return paired_bases, unpaired_bases

def writeSummary(seq_id, iteration, score, pred, fname):
    ls = [seq_id, iteration, score, pred]
    with open(fname, 'a') as myfile:
        writer_object = writer(myfile)
        writer_object.writerow(ls)