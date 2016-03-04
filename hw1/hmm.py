# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
from collections import Counter
import numpy as np
from scipy.misc import logsumexp
from scipy.sparse import dok_matrix, dia_matrix
np.seterr(under='warn')

OFFSET = 5
THRESHOLD = -3
EMITR = 10
TODECODE = 150
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

  engwords = Counter()
  for se in engcorp: 
    engwords.update(se)
  eng_w2c = {}
  #eng_c2w = []
  c=0
  for e, ct in engwords.most_common(len(engwords)): 
    eng_w2c[e]=c 
    #eng_c2w[c]=e
    c+=1
  frawords = Counter()
  for sf in fracorp: 
    frawords.update(sf)
  fra_w2c = {}
  #frac2w = []
  c=0
  for f, ct in frawords.most_common(len(frawords)):
    fra_w2c[f]=c
    #fra_c2w[c]=f
    c+=1

  for i in xrange(len(engcorp)): 
    engcorp[i] = [eng_w2c[e] for e in engcorp[i]]
  for i in xrange(len(fracorp)): 
    fracorp[i] = [fra_w2c[f] for f in fracorp[i]]

  engvocab = len(engwords)
  fravocab = len(frawords)
  print fracorp[385], franum[385], fra_w2c['.']

  assert len(engcorp)== len(fracorp)== len(engnum)== len(franum)
  return engcorp, fracorp, engnum, franum, engvocab, fravocab

def e2f(engcorp, fracorp, t):
  count = defaultdict(lambda: defaultdict(float))
  total = defaultdict(float)

  for sitr in range(opts.num_sents):
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

# prints alignments 
def decode_viterbi(mu, franum, engnum): 
  alignment = mu.argmax(axis=1)
  for i in xrange(alignment.shape[0]): 
    print str(franum[i])+"-"+str(engnum[alignment[i]])+" ", 
  print ""
def decode_posterior(mu, franum, engnum, thr): 
  alignment = np.where(mu > thr)
  for x, y in alignment: 
    print str(franum[x])+"-"+str(engnum[y])+" ", 
  print ""

# SOURCE: MICHAEL COLLINS NOTES
# len prevain, prevaout = OFFSET, len preva = 2*OFFSET+1
def fb(engsent, frasent, t, tx, engfirst, prevain, prevaout, preva, prevainx, prevaoutx, prevax):
  m, n = len(frasent), len(engsent)
  # emit[fra][eng] emit fra given eng
  emit = np.ndarray(shape=(m, n), dtype=np.float)
  for j in xrange(m):
    for i in xrange(n):
      emit[j][i] = t[(engsent[i],frasent[j])]
  ###np.divide(emit, emit.sum(axis=0), emit)
  emit = emit - logsumexp(emit, axis=0)
  
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

  distor = np.log(distor)
  distor_in = np.log(distor_in)
  distor_out = np.log(distor_out)

  # psi[j][s][sp]
  ###psi = np.multiply(np.expand_dims(emit, axis=2), np.expand_dims(distor, axis=0))
  psi = np.logaddexp(np.expand_dims(emit, axis=2), np.expand_dims(distor, axis=0))

  #print emit
  # Initialize alpha 
  alpha = np.zeros(shape=(m, n), dtype=np.float)
  alpha[0] = distor_in
  # FORWARD 
  for j in xrange(1, m):
    ###alpha[j] = np.multiply(psi[j], alpha[j-1]).sum(axis=1)
    alpha[j] = logsumexp(np.logaddexp(psi[j], alpha[j-1]), axis=1)

  # Initialize beta
  beta = np.zeros(shape=(m, n), dtype=np.float)
  beta[m-1] = distor_out
  # BACKWARD 
  for j in xrange(m-2, -1, -1): 
    ###beta[j] = np.multiply(psi[j+1], beta[j+1]).sum(axis=1)
    beta[j] = logsumexp(np.logaddexp(psi[j+1], beta[j+1]), axis=1)

  ###Z = np.multiply(distor_out, alpha[m-1]).sum()
  Z = logsumexp(np.logaddexp(distor_out, alpha[m-1]))
  # mu[j][a]: j is a
  ###mu = np.multiply(alpha, beta)
  ###np.divide(mu, Z, mu)
  mu = np.add(alpha, beta)
  np.subtract(mu, Z, mu)

  
  # mu3[j][a][b]: j is a, j+1 is b
  ###mu3 = np.multiply(psi[1:], np.expand_dims(alpha[:-1], axis=1))
  ###np.multiply(mu3, np.expand_dims(beta[1:], axis=2), mu3)
  ###np.divide(mu3, Z, mu3)
  if m > 1: 
      mu3 = np.add(psi[1:], np.expand_dims(alpha[:-1], axis=1))
      np.add(mu3, np.expand_dims(beta[1:], axis=2), mu3)
      np.subtract(mu3, Z, mu3)

  # mu3_in[a]: first position is a
  ###mu3_in = np.divide(beta[0], beta[0].sum())
  ###mu3_out = np.divide(alpha[m-1], alpha[m-1].sum())
  mu3_in = np.subtract(beta[0], logsumexp(beta[0]))
  mu3_out = np.subtract(alpha[m-1], logsumexp(alpha[m-1]))

  # Accumulate translation probs 
  for j in xrange(m):
    for i in xrange(n):
      ###tx[(engsent[i],frasent[j])] += mu[j][i]
      tx[(engsent[i],frasent[j])] = np.logaddexp(tx[(engsent[i],frasent[j])], mu[j][i])

  # Accumulate transition probs 
  # line 1
  for i in xrange(n): 
    ###prevainx[min(i,OFFSET-1)] += mu3_in[i]
    prevainx[min(i,OFFSET-1)] += np.exp(mu3_in[i])

  # line bulk 
  ###p = mu3.sum(axis=0) # do not normalize
  if m > 1: 
      p = logsumexp(mu3, axis=0)
      for i in xrange(n):
        for j in xrange(n): 
          ###prevax[min(max(-OFFSET,j-i),OFFSET)+OFFSET] += p[i][j]
          prevax[min(max(-OFFSET,j-i),OFFSET)+OFFSET] += np.exp(p[i][j])

  # line last 
  for i in xrange(n):
    ###prevaoutx[min(i,OFFSET-1)] += mu3_out[i]
    prevaoutx[min(i,OFFSET-1)] += np.exp(mu3_out[i])
  return mu

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with HMM1...\n")
bitext = [[sentence.strip().lower().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
sys.stderr.write("Preparing...\n")
engcorp, fracorp, engnum, franum, engvocab, fravocab = prep(bitext)
sys.stderr.write("ENG vocab size: "+str(engvocab)+"\n")
sys.stderr.write("FRA vocab size: "+str(fravocab)+"\n")

# Initialize tef with model 1 
tef = defaultdict(lambda: defaultdict(lambda: float(1)))
for itr in range(4): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  tef = e2f(engcorp, fracorp, tef)

# initialize buckets 
prevain = np.arange(OFFSET, 0, -1) * 1.0 / np.arange(OFFSET, 0, -1).sum()
prevaout = np.arange(OFFSET, 0, -1) * 1.0 / np.arange(OFFSET, 0, -1).sum()
preva = np.empty(2*OFFSET+1)
preva[:OFFSET] = np.arange(1, OFFSET+1) * 1.0 
preva[OFFSET:] = np.arange(OFFSET+1, 0, -1) * 1.0
preva = preva / preva.sum()

t = dok_matrix((engvocab,fravocab),dtype=float)
for e in tef: 
  for f in tef[e]: 
    ### t[(e,f)] = tef[e][f]
    t[(e,f)] = np.log(tef[e][f])
tef = t

sys.stderr.write("EMing...\n")
for itr in xrange(EMITR): 
  sys.stderr.write("itr: "+str(itr)+"\n")
  # Initializing counters 
  prevainx = np.zeros(shape=(OFFSET,),dtype=float)
  prevaoutx = np.zeros(shape=(OFFSET,),dtype=float)
  prevax = np.zeros(shape=(2*OFFSET+1,),dtype=float)
  tefx = dok_matrix((engvocab,fravocab),dtype=float)

  # E-step
  poster = []
  for x in xrange(opts.num_sents): 
    p = fb(engcorp[x], fracorp[x], tef, tefx, True, prevain, prevaout, preva, prevainx, prevaoutx, prevax)
    poster.append(p)

  # M-step
  tefx = tefx.tocoo().tocsc()
  t_save = dia_matrix((np.reciprocal(tefx.sum(axis=0).flatten()),[0,]), shape=(fravocab, fravocab))
  tef = tefx.dot(t_save).todok()

  np.divide(prevainx,prevainx.sum(), prevain)
  np.divide(prevaoutx,prevaoutx.sum(), prevaout)
  np.divide(prevax,prevax.sum(), preva)

  #sys.stderr.write(str(tef)+"\n")
  sys.stderr.write(str(preva)+str(preva.sum())+"\n")
  sys.stderr.write(str(prevain)+str(prevain.sum())+"\n")
  sys.stderr.write(str(prevaout)+str(prevaout.sum())+"\n")


sys.stderr.write("Decoding...\n")
for itr in xrange(TODECODE):
  #decode_posterior(poster[itr], franum[itr], engnum[itr], THRESHOLD)
  decode_viterbi(poster[itr], franum[itr], engnum[itr])
