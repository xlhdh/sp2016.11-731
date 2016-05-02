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
	cabvo = {}
	for i, (word, _) in enumerate(counted.most_common()): 
		vocab[word] = i + 2
		cabvo[i+2] = word
	vocab['<s>'] = i + 3
	cabvo[i+3] = '<s>'
	vocab['</s>'] = 0
	cabvo[0] = '</s>'
	vocab['<unk>'] = 1
	cabvo[1] = '<unk>'

	return vocab, cabvo

from blocks.extensions import SimpleExtension 
class Sampler(SimpleExtension):
	def __init__(self, model, data_stream, vocab, saveto, **kwargs):
		super(Sampler, self).__init__(**kwargs)
		self.model = model
		self.data_stream = data_stream
		self.sampling_fn = model.get_theano_function()
		self.vocab = vocab
		self.saveto = saveto

	def do(self, which_callback, *args):
		# Sample
		print ''
		fo = open(self.saveto+'.txt', 'w')
		for batch in self.data_stream.get_epoch_iterator(): 
			inputs = batch[0]
			outputs = self.sampling_fn(inputs)[0]
			for i in xrange(outputs.shape[0]): 
				eside = outputs[i]
				s =  u''.join([self.vocab[j]+u' ' for j in eside])+'\n'
				fo.write(s.encode('utf8'))

class Plotter(SimpleExtension):
	def __init__(self, saveto, **kwargs):
		super(Plotter, self).__init__(**kwargs)
		self.saveto = saveto

	def do(self, which_callback, *args):
		fo = open(self.saveto+'.txt', 'a')
		log = self.main_loop.log
		for key, value in log.current_row.items():
			fo.write(str(value)+' ')
		fo.write('\n')
