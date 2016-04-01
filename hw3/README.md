Used DP to compute best hypotheses produce f[i:j] of the input sentence. (Like Austin did, but I had the idea before looking at his Readme!)

Tried recombination by max probability or by sum of probabilities for the translation string. This sum is an approximation to the true sum, and it seems to work better (than pure max) when the stack size is small. 

Tried recombination based on translation string or just the first 2 and last 2 words of the translation string ("states"). 

During my implementation I had a bug that only scores the LM probability for the first 3 words of the English translation (I would assume the majority of the English phrases are indeed 3 words or less, but some of them are not). This worked extremely well. I guess this is a way of 'pruning' phrases that are too long that they tend to have lower probablities? 

It is not possible to compute the true LM prob of the first 2 grams of the hyotheses of a f[i:j] span, so I used the LM prob over an empty state as 'future cost' and deducted this when I stitch the hyothesis together. 

I used the grading script to score the n-best list. 

I wrote a (supposingly) decoder that prunes over true probablities, but with stack size 5-10 or less it does not work as well as the above with s=~100, and it takes forever to run if the stack size is larger. 

Finally, I downloaded Austin's pseudo-oracle and asserted that my output was lower or equal to his on each sentence :(. I then picked sentences that had large descrepencies (they tend to be longer sentences) and used multiprocessing to decode them with large stack sizes (~300-500). Unfortunately this leads to only ~2 points improments overall, so the problem is probably at my pruning techniques instead of the stack size. 

The majority of the final results are obtained with `./aproto-sum_copy -s 100`. 


There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

