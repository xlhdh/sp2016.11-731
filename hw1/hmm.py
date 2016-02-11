# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np
OFFSET = 2
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

# SOURCE: MICHAEL COLLINS NOTES
# len prevain, prevaout = OFFSET, len preva = 2*OFFSET+1
def fb(engsent, frasent, t, tx, engfirst, prevain, prevaout, preva, prevainx, prevaoutx, prevax):
  m, n = len(frasent), len(engsent)
  # emit[eng][fra]
  emit = np.ndarray(shape=(n, m), dtype=np.float)
  for i in xrange(n):
    for j in xrange(m):
      emit[i][j] = t[engsent[i]][frasent[j]]
  np.divide(emit, emit.sum(axis=0), emit)

  # distor[cur][prev]
  distor = np.ndarray(shape=(n, n), dtype=np.float) 
  for i in xrange(n):
    for j in xrange(n):
      distor[i][j]=preva[ min(max(-OFFSET,i-j),OFFSET)+OFFSET ] # i-j: cur-prev
  np.divide(distor, distor.sum(axis=0), distor)

  distor_in = np.ndarray(shape=(n,), dtype=np.float) 
  for i in xrange(n):
    distor_in[i] = prevain[min(i,OFFSET-1)]
  np.divide(distor_in, distor_in.sum(), distor_in)

  distor_out = np.ndarray(shape=(n,), dtype=np.float) 
  for i in xrange(n):
    distor_out[i] = prevain[min(n-1-i,OFFSET-1)]
  np.divide(distor_out, distor_out.sum(), distor_out)

  # psi[j][s][sp]
  psi = np.multiply(np.expand_dims(emit, axis=2), distor)

  print emit
  # Initialize alpha 
  alpha = np.zeros(shape=(m, n), dtype=np.float)
  alpha[0] = distor_in
  # FORWARD 
  for j in xrange(1, m):
    alpha[j] = np.multiply(psi[j], alpha[j-1]).sum(axis=1)

  # Initialize beta
  beta = np.zeros(shape=(m, n), dtype=np.float)
  beta[m-1] = distor_out
  # BACKWARD 
  for j in xrange(m-2, -1, -1): 
    beta[j] = np.multiply(psi[j+1], beta[j+1]).sum(axis=1)

  Z = np.multiply(distor_out, alpha[m-1]).sum()
  # mu[j][a]: j is a
  mu = np.multiply(alpha, beta)
  np.divide(mu, Z, mu)
  
  # mu3[j][a][b]: j is a, j+1 is b
  mu3 = np.multiply(psi[1:], np.expand_dims(alpha[:-1], axis=1))
  np.multiply(mu3, np.expand_dims(beta[1:], axis=2), mu3)
  np.divide(mu3, Z, mu3)

  # mu3_in[a]: first position is a
  mu3_in = np.divide(beta[0], beta[0].sum())
  mu3_out = np.divide(alpha[m-1], alpha[m-1].sum())

  # Accumulate translation probs 
  for s in xrange(n):
    for j in xrange(m): 
      tx[engsent[i]][frasent[j]] += mu[j][i]

  # Accumulate transition probs 

  # line 1
  for i in xrange(n):
    prevainx[min(i,OFFSET-1)] += mu3_in[i]

  # line bulk 
  p = mu3.sum(axis=0) # do not normalize
  for i in xrange(n):
    for j in xrange(n): 
      prevax[min(max(-OFFSET,j-i),OFFSET)+OFFSET] += p[i][j]

  # line last 
  for i in xrange(n):
    prevaoutx[min(i,OFFSET-1)] += mu3_out[i]


  


optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/small_dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM1...\n")
bitext = [[sentence.strip().lower().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
sys.stderr.write("Preparing...\n")
engcorp, fracorp, engnum, franum = prep(bitext)

t = defaultdict(lambda: defaultdict(lambda: float(1)))
tx = defaultdict(lambda: defaultdict(lambda: float(0)))
prevain, prevaout, preva = [0.5,0.5], [0.5,0.5], [0.2,0.2,0.2,0.2,0.2]
for itr in xrange(4): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  prevainx, prevaoutx, prevax = [0.0,0.0], [0.0,0.0], [0.0,0.0,0.0,0.0,0.0]
  for x in xrange(len(engcorp)): 
    fb(engcorp[x], fracorp[x], t, tx, True, prevain, prevaout, preva, prevainx, prevaoutx, prevax)
  txf = defaultdict(float)
  for e in tx:
    for f in tx[e]: 
      txf[f]+= tx[e][f]
  for e in tx:
    for f in tx[e]: 
      print e, f
      t[e][f] = tx[e][f]/txf[f]
  for i in xrange(OFFSET): 
    prevain[i] = prevain[i]/sum(prevainx)
  for i in xrange(OFFSET): 
    prevaout[i] = prevaout[i]/sum(prevaoutx)
  for i in xrange(2*OFFSET+1): 
    preva[i] = preva[i]/sum(prevax)
  print prevain, prevaout, preva


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


sys.stderr.write("Decoding...\n")
todecode = 150
dcdef = dcd(engcorp[:todecode], fracorp[:todecode], engnum[:todecode], franum[:todecode], tef)
dcdfe = dcd(fracorp[:todecode], engcorp[:todecode], franum[:todecode], engnum[:todecode], tfe)
dcdfe = [[(y, x) for x, y in l] for l in dcdfe]

#alm  = [set(x).union(set(y)) for x, y in zip(dcdef, dcdfe)]
#alm  = [set(x).intersection(set(y)) for x, y in zip(dcdef, dcdfe)]
alm = dcdef

for i in range(len(alm)): 
  sys.stdout.write(' '.join([str(x[0])+'-'+str(x[1]) for x in alm[i]])+'\n')


