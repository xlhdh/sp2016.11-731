This classifier is based on sklearn's multiclass classification implementation. 
#### First, I prepared the following features: 

Precision, stemmed and row, 1-2 grams. 
Recall, stemmed and raw, 1-3 grams. 

Precision and recall, character-wise (credit: Austin), 1-7 grams. 

Sentence distance, cosine and euclidean (credit: Chris/Austin). 
note: this metric stacks the GloVe representation of each word in a sentence, and then multiplies the metrix by its own transpose to form a D by D matrix, where D is the lenth of the GloVe vector. Then the vector is flattened and we can compute the dictance between the hypothesis and the ref. 

Language model scores from 3 models: 
2 3-gram models from https://keithv.com/software/giga/. 
1 other 4-gram model trained off of the Europarl corpus. 
The software used to make/query language models is https://kheafield.com/code/kenlm/. 

#### Second, I used a "greedy search" to find which features should be included in my classifier. 
I started with only 1 feature, pick the one with the highest cross-validation score (and low variance), and then add 1 more feature at a time. 
I ended up picking Chracter-level recall 4-gram, sentence distance cosine, language model Europarl, ???, in that order. It's easy to see that they are features will few correlations with each other. 
In terms of classifiers, I tried SVM with different kernals, logistic regression, naive Bayes, etc. They are all part of the greedy search process - I search for the right classifier and the feature set at the same time. 

#### Others 
I flipped the training data to form a second copy. At first, I thought there is no point to train the classifier to prefer h1 over h2. I briefly deviated to removing this constraint thinking human may be biased on the position a sentence appear on the screen but soon gave up as I saw no improvement training on only one copy of the data. 
I used the raw scores of the two hyps instead of their different to allow the model to learn some interactions with different scores if it wants to. It can simply take a 45 degree dicision surface if it decides that the score themselves are not relevant. 
Good preprocessing (that is a personalized tokenizer rathe than the NLTK default, etc.) gave me around 1% of the result, which is siginificant for this task. 


There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.
