#!/usr/bin/env python
# -*- coding: utf-8“ -*-

import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
import numpy as np
import string
 
def gloves(): 
    glovefile = '/Users/hyz/Desktop/glove.840B.300d.txt' 
    with open(glovefile, 'r') as fl: 
        for ln in fl: 
            yield ln
def sentences():
    toker = RegexpTokenizer('[^​​“”…\\\\%s\s]+|[​​“”…\\\\%s]'.decode('utf8') % (string.punctuation,string.punctuation))
    with open('data/train-test.hyp1-hyp2-ref', 'r') as f:
        for pair in f:
            yield [toker.tokenize(sentence.decode('utf8').lower()) for sentence in pair.split(' ||| ')]
   

def main():
    ctr = Counter()
    for a, b, c in sentences(): 
        ctr.update(a)
        ctr.update(b)
        ctr.update(c)
    mywords = ctr.keys()
    '''for w, c in ctr.most_common(16727):
        print w, c'''

    for ln in gloves(): 
        if ln.decode('utf8').split()[0].lower() in mywords: 
            print ln,

# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
