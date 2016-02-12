# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np
OFFSET = 5
THRESHOLD = 0.5
EMITR = 20
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
    #engcorp.append(se+["_NULL_",])
    engcorp.append(se)
    #engnum.append(range(len(se))+[None,])
    engnum.append(range(len(se)))


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
    #fracorp.append(fline+["_NULL_",])
    fracorp.append(fline)
    #franum.append(fnum+[None,])
    franum.append(fnum)

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
  # emit[fra][eng] emit fra given eng
  emit = np.ndarray(shape=(m, n), dtype=np.float)
  for i in xrange(m):
    for j in xrange(n):
      emit[i][j] = t[engsent[j]][frasent[i]]
  if np.count_nonzero(emit.sum(axis=0))-np.prod(emit.shape[1]):
    print emit
    print engsent, frasent
  np.divide(emit, emit.sum(axis=0), emit)

  # Constructing Distortion distributions 
  # First Line distor_in[a] where the (first alignment - 0) is at 'a'
  distor_in = np.zeros(shape=(n,), dtype=np.float) 
  if n > OFFSET: 
    distor_in[:OFFSET-1] = prevain[:OFFSET-1]
    distor_in[OFFSET-1:] += prevain[-1]/(n-OFFSET+1)
  else: 
    distor_in = prevain[:n]
  # bulk lines distor[cur][prev] where 
  distor = np.zeros(shape=(n,n), dtype=np.float)
  for cur in xrange(n): 
    # range of cur-prev is [cur-(n-1), cur-0] 
    if cur-(n-1) <= -OFFSET:  # left padding: if more than 1 tail on left
      distor[cur][cur+OFFSET:] += preva[0]/(n-cur-OFFSET) #share the left tail 
    if cur-0 >= OFFSET: 
      distor[cur][:cur-OFFSET+1] = preva[-1]/(cur-OFFSET+1) #share the right tail 
    for prev in xrange(n): 
      if -OFFSET < cur-prev < OFFSET:
        distor[cur][prev] = preva[cur-prev]
  # Last line distor_out[a] where (n-1  - last alignment) is at 'a'
  distor_out = np.zeros(shape=(n,), dtype=np.float) 
  if n> OFFSET: 
    distor_out[:OFFSET-1] = prevaout[:OFFSET-1]
    distor_out[OFFSET-1:] += prevaout[-1]/(n-OFFSET+1)
  else: 
    distor_out = prevaout[:n]
  
  np.divide(distor, distor.sum(axis=1), distor)
  np.divide(distor_in, distor_in.sum(), distor_in)
  np.divide(distor_out, distor_out.sum(), distor_out)

  # psi[j][s][sp]
  psi = np.multiply(np.expand_dims(emit, axis=2), np.expand_dims(distor, axis=0))

  #print emit
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
  for j in xrange(m):
    for i in xrange(n):
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
  return mu


optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with HMM1...\n")
bitext = [[sentence.strip().lower().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
sys.stderr.write("Preparing...\n")
engcorp, fracorp, engnum, franum = prep(bitext)

tef = defaultdict(lambda: defaultdict(lambda: float(1)))
for itr in range(4): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  tef = e2f(engcorp, fracorp, tef)

prevain = np.empty(OFFSET)
prevain.fill(1.0/OFFSET)
prevain = np.arange(OFFSET, 0, -1) * 1.0 / np.arange(OFFSET, 0, -1).sum()
prevaout = np.empty(OFFSET)
prevaout.fill(1.0/OFFSET)
prevaout = np.arange(OFFSET, 0, -1) * 1.0 / np.arange(OFFSET, 0, -1).sum()
preva = np.empty(2*OFFSET+1)
preva.fill(1.0/(2*OFFSET+1))
preva[:OFFSET] = np.arange(1, OFFSET+1) * 1.0 
preva[OFFSET:] = np.arange(OFFSET+1, 0, -1) * 1.0
preva = preva / preva.sum()

sys.stderr.write("EMing...\n")
for itr in xrange(ENITR): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  #sys.stderr.write(str(tef)+"\n")
  prevainx = np.zeros(shape=(OFFSET,),dtype=float)
  prevaoutx = np.zeros(shape=(OFFSET,),dtype=float)
  prevax = np.zeros(shape=(2*OFFSET+1,),dtype=float)
  tefx = defaultdict(lambda: defaultdict(lambda: float(0)))
  poster = []
  for x in xrange(len(engcorp)): 
    p = fb(engcorp[x], fracorp[x], tef, tefx, True, prevain, prevaout, preva, prevainx, prevaoutx, prevax)
    poster.append(p)
  txf = defaultdict(float)
  for e in tefx:
    for f in tefx[e]: 
      txf[f]+= tefx[e][f]
  for e in tefx:
    for f in tefx[e]: 
      tef[e][f] = tefx[e][f]/txf[f]
      if txf[f] == 0.0:
        sys.stderr.write(e+"-"+f+"\n")
  np.divide(prevainx,prevainx.sum(), prevain)
  np.divide(prevaoutx,prevaoutx.sum(), prevaout)
  np.divide(prevax,prevax.sum(), preva)

  #sys.stderr.write(str(tef)+"\n")
  sys.stderr.write(str(preva)+str(preva.sum())+"\n")
  sys.stderr.write(str(prevain)+str(prevain.sum())+"\n")
  sys.stderr.write(str(prevaout)+str(prevaout.sum())+"\n")
sys.stderr.write("ENG vocab size: "+str(len(tef))+"\n")


sys.stderr.write("Decoding...\n")
todecode = 100
for itr in xrange(todecode):
  poste = poster[itr]
  #print poster[itr].shape
  #print len(fracorp[itr]), len(engcorp[itr])
  for j in xrange(poste.shape[0]): # -1 for _NULL_
    for i in xrange(poste.shape[1]):
      if poste[j][i] > THRESHOLD: 
         print str(franum[itr][j])+"-"+str(engnum[itr][i])+" ", 
  print ""

#    opt = np.vstack(np.where(poste > THRESHOLD)).T
#    print " ".join([str(opt[i][0])+"-"+str(opt[i][1]) for i in xrange(opt.shape[0])])
  
#dcdef = dcd(engcorp[:todecode], fracorp[:todecode], engnum[:todecode], franum[:todecode], tef)
#dcdfe = dcd(fracorp[:todecode], engcorp[:todecode], franum[:todecode], engnum[:todecode], tfe)
#dcdfe = [[(y, x) for x, y in l] for l in dcdfe]

#dcdhmm = dcd(engcorp[:todecode], fracorp[:todecode], engnum[:todecode], franum[:todecode], tef)

#alm  = [set(x).union(set(y)) for x, y in zip(dcdef, dcdfe)]
#alm  = [set(x).intersection(set(y)) for x, y in zip(dcdef, dcdfe)]
#alm = dcdhmm

#for i in range(len(alm)): 
#  sys.stdout.write(' '.join([str(x[0])+'-'+str(x[1]) for x in alm[i]])+'\n')


