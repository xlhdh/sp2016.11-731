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

from blocks.bricks.recurrent import GatedRecurrent, LSTM
from blocks.bricks import Tanh, MLP
from blocks.bricks.base import application
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans
class GRU2GO(GatedRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(GRU2GO, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer.apply(
            attended[0, :, -self.attended_dim:])
        return initial_state

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)

class LSTM2GO(LSTM):
    def __init__(self, attended_dim, **kwargs):
        super(LSTM2GO, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer_s = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer_s)

        self.initial_transformer_c = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='cell_initializer')
        self.children.append(self.initial_transformer_c)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = self.initial_transformer_s.apply(
            attended[0, :, -self.attended_dim:])
        initial_cell = self.initial_transformer_c.apply(
            attended[0, :, -self.attended_dim:])
        return [initial_state, initial_cell]

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget, self.W_cell_to_out]

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
			outputs = self.sampling_fn(inputs)[0].T
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
		log = self.main_loop.log.current_row
		fo.write(str(log['tra_target_cost'])+' '+str(log['dev_target_cost'])+'\n')

		
