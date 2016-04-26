# Helper class
from blocks.bricks import FeedforwardSequence, Initializable
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass

def text_to_dict(files): 
	from collections import Counter
	'''Input: list of file names
	   Output: dict of vocabulary '''
	counted = Counter() 
	for fname in files: 
		with open(fname, 'r') as f: 
			for line in f: 
				counted.update(line.decode('utf8').split())
	sp = ['<s>','</s>','<unk>']
	for s in sp: 
		del counted[s]
	vocab = {} 
	for i, (word, _) in enumerate(counted.most_common()): 
		vocab[word] = i + 2
	vocab['<s>'] = i + 3
	vocab['</s>'] = 0
	vocab['<unk>'] = 1

	return vocab