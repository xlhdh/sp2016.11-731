Description of project: 
    python ibm1.py -n 100000 | ./check | ./grade -n 5
gives the output.txt . 

Techniques used in addition to IBM1: 
- German words are splitted according to jwordsplitter-4.1.jar
- Model trained with both direction, taking the intersection (this works better than union or single model)
- identical words are unconditionally aligned (credit: Qinlan)

The hmm.py file implements the HMM model discribed in Vogel, Ney, Tillman 1996, with the several features (3 sets of distortion buckets, initializing with EM transtion tables and diagnal distortion parameters, threshold decoding, etc.) in Liang, Taskar, Klein 2006. But it didn't work. 


There are three Python programs here (`-h` for usage):

 - `./align` aligns words using Dice's coefficient.
 - `./check` checks for out-of-bounds alignment points.
 - `./grade` computes alignment error rate.

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./align -t 0.9 -n 1000 | ./check | ./grade -n 5


The `data/` directory contains a fragment of the German/English Europarl corpus.

 - `data/dev-test-train.de-en` is the German/English parallel data to be aligned. The first 150 sentences are for development; the next 150 is a blind set you will be evaluated on; and the remainder of the file is unannotated parallel data.

 - `data/dev.align` contains 150 manual alignments corresponding to the first 150 sentences of the parallel corpus. When you run `./check` these are used to compute the alignment error rate. You may use these in any way you choose. The notation `i-j` means the word at position *i* (0-indexed) in the German sentence is aligned to the word at position *j* in the English sentence; the notation `i?j` means they are "probably" aligned.

