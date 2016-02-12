# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np
# GLOBAL t is t[engword][fraword]

def prep(bitext):
  engcorp, fracorp, engnum, franum = [], [], [], []
  fradict = {}
  with open("gdrdict.txt") as fl: 
    for ln in fl: 
      ln = ln[:-1].split('=')
      fradict[ln[0]]=ln[1].split('+')
  for (sf, se) in bitext: 
    # ENG side
    engcorp.append(se+["_NULL_",])
    engnum.append(range(len(se))+[None,])

    # FRA side 
    fline = []
    fnum = []     
    for (i, fword) in enumerate(sf): 
      fword = fword.strip()
      if any(c.isdigit() for c in fword): 
        fword = [fword, ]
      else: 
        fword = fword.replace("„","").split("-")
      fout = []
      for f in fword: 
        if f in fradict: 
          fout.extend(fradict[f])
        else: 
          fout.extend([f,])
      fline.extend(fout)
      fnum.extend([i,]*len(fout))

    try: 
      while True: 
        i = fline.index("")
        del fline[i]
        del fnum[i]
    except ValueError: 
      pass
    assert len(fline)==len(fnum)
    fracorp.append(fline+["_NULL_",])
    franum.append(fnum+[None,])

  assert len(engcorp)== len(fracorp)== len(engnum)== len(franum)
  return engcorp, fracorp, engnum, franum 

def e2f(engcorp, fracorp, t):
  count = defaultdict(lambda: defaultdict(float))
  total = defaultdict(float)

  for sitr in range(len(engcorp)):
    se = engcorp[sitr]
    sf = fracorp[sitr]
    s = defaultdict(float)
    for e in se:
      for f in sf: 
        s[e] += t[e][f]
    for e in se: 
      for f in sf: 
        count[e][f] += t[e][f]/s[e]
        total[f] += t[e][f]/s[e]
  
  for e in t.keys():
    for f in t[e].keys():
      t[e][f] = count[e][f]/total[f]
  return t

def dcd(engcorp, fracorp, engnum, franum, tef):
  opt  = []
  for itr in range(len(engcorp)):
    optline = []
    se = engcorp[itr]
    sf = fracorp[itr]
    for (i, e) in enumerate(se): 
      x = franum[itr][np.argmax([tef[e][f] for f in sf])], engnum[itr][i]
      if x[0] and x[1]:
        optline.append(x)
    opt.append(optline)
  return opt



optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM1...\n")
bitext = [[sentence.strip().lower().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
sys.stderr.write("Preparing...\n")
engcorp, fracorp, engnum, franum = prep(bitext)

sys.stderr.write("EMing...\n")
tef = defaultdict(lambda: defaultdict(lambda: float(1)))
for itr in range(4): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  tef = e2f(engcorp, fracorp, tef)
sys.stderr.write("ENG vocab size: "+str(len(tef))+"\n")
tfe = defaultdict(lambda: defaultdict(lambda: float(1)))
for itr in range(4): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  tfe = e2f(fracorp, engcorp, tfe)
sys.stderr.write("FRA vocab size: "+str(len(tfe))+"\n")

#import cPickle as pkl
#pkl.dump(tef, open("tef.pkl", 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
#pkl.dump(tfe, open("tfe.pkl", 'wb'), protocol=pkl.HIGHEST_PROTOCOL)

sys.stderr.write("Decoding...\n")
todecode = len(engcorp)
dcdef = dcd(engcorp[:todecode], fracorp[:todecode], engnum[:todecode], franum[:todecode], tef)
dcdfe = dcd(fracorp[:todecode], engcorp[:todecode], franum[:todecode], engnum[:todecode], tfe)
dcdfe = [[(y, x) for x, y in l] for l in dcdfe]

alm = dcdef
fn = open("outef.txt", 'w')
for i in range(len(alm)): 
  fn.write(' '.join([str(x[0])+'-'+str(x[1]) for x in alm[i]])+'\n')
fn.close()

alm = dcdfe
fn = open("outfe.txt", 'w')
for i in range(len(alm)): 
  fn.write(' '.join([str(x[0])+'-'+str(x[1]) for x in alm[i]])+'\n')
fn.close()

alm = [set(x).intersection(set(y)) for x, y in zip(dcdef, dcdfe)]
fn = open("outinter.txt", 'w')
for i in range(len(alm)): 
  fn.write(' '.join([str(x[0])+'-'+str(x[1]) for x in alm[i]])+'\n')
fn.close()

alm = [set(x).union(set(y)) for x, y in zip(dcdef, dcdfe)]
fn = open("outuni.txt", 'w')
for i in range(len(alm)): 
  fn.write(' '.join([str(x[0])+'-'+str(x[1]) for x in alm[i]])+'\n')
fn.close()

