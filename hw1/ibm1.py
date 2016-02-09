#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM1...\n")
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]

'''
for (e, f) in bitext: 
  e = e.extend("_NULL_")
  f = f.extend("_NULL_")
'''

def e2f(thebitext):
  t = defaultdict(lambda: defaultdict(lambda: float(1)))
  for itr in range(6):
    sys.stderr.write("itr: "+str(itr)+"\n")
    count = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)
    for (sf, se) in thebitext:
      s = defaultdict(float)
      for e in se:
        for f in sf: 
          s[e] += t[e][f]
      for e in se: 
        for f in sf: 
          count[e][f] += t[e][f]/s[e]
          total[f] += t[e][f]/s[e]
    
    for f in total.keys():
      for e in t.keys():
        if f in t[e]:
          t[e][f] = count[e][f]/total[f]
  return t
def f2e(thebitext):
  t = defaultdict(lambda: defaultdict(lambda: float(1)))
  for itr in range(6):
    sys.stderr.write("itr: "+str(itr)+"\n")
    count = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)
    for (se, sf) in thebitext:
      s = defaultdict(float)
      for e in se:
        for f in sf: 
          s[e] += t[e][f]
      for e in se: 
        for f in sf: 
          count[e][f] += t[e][f]/s[e]
          total[f] += t[e][f]/s[e]
    
    for f in total.keys():
      for e in t.keys():
        if f in t[e]:
          t[e][f] = count[e][f]/total[f]
  return t


tef = e2f(bitext)
tfe = f2e(bitext)

for (sf, se) in bitext[:150]:
  for (i, e) in enumerate(se): 
      sys.stdout.write(str(np.argmax([tef[e][f] for f in sf]))+"-"+str(i)+" ")
  for (i, f) in enumerate(sf): 
      sys.stdout.write(str(i)+"-"+str(np.argmax([tfe[f][e] for e in se]))+" ")
  sys.stdout.write("\n")

