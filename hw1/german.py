 # -*- coding: UTF-8 -*-
 #!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import re


optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write("Training with IBM1...\n")
bitext = [[sentence.lower().strip().split() for sentence in pair.split(' ||| ')][0] for pair in open(opts.bitext)][:opts.num_sents]

words = []
for sf in bitext:
  words.extend(sf)

words = set(words)

"„"

for w in words:
  if '-' in w and not any(c.isdigit() for c in w):
    #print w+'=', 
    for u in w.split('-'):
      print u
  else:
    print w



#/\b([äöüÄÖÜß\w]+)\b/g