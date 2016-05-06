LSTM encoder - decoder with word embedding layer on both sides, and 1 layer LSTM on both sides. 
Tried: 
- various embedding and hidden layer sizes 
- various batch sizes 
- gradually increasing sequence length 
- initializing forget gates bias with 1.0 
Nothing worked (or to be precise, they all worked, just not enough to make this a decent translator). Final result is from  embedding size 128, hidden layer 512, and batch size 32. 

There are two Python programs here:

 - `python bleu.py your-output.txt ref.txt` to compute the BLEU score of your output against the reference translation.
 - `python rnnlm.py ref.txt` trains an LSTM language model, just for your reference if you want to use pyCNN to perform this assignment.

The `data/` directory contains the files needed to develop the MT system:

 - `data/train.*` the source and target training files.

 - `data/dev.*` the source and target development files.

 - `data/test.src` the source side of the blind test set.
