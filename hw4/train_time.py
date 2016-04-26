import configure
from collections import Counter
from toolz import merge

from util import text_to_dict, InitializableFeedforwardSequence
from stream import get_src_stream, get_src_tgt_stream

from theano import tensor
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant

from blocks.bricks.recurrent import GatedRecurrent, LSTM, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.sequence_generators import (
	SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks import Tanh, Bias, Linear, Maxout

from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Load, Checkpoint

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, CompositeRule)


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def printchildren(parent, i):
	print '<....>'*i, parent.name
	for child in parent.children: 
		printchildren(child, i+1)


def main(config): 
	vocab_src = text_to_dict([config['train_src'],
		config['dev_src'], config['test_src']])
	vocab_tgt = text_to_dict([config['train_tgt'],
		config['dev_tgt']])

	# Create Theano variables
	logger.info('Creating theano variables')
	source_sentence = tensor.lmatrix('source')
	source_sentence_mask = tensor.matrix('source_mask')
	target_sentence = tensor.lmatrix('target')
	target_sentence_mask = tensor.matrix('target_mask')
	source_sentence_smp = tensor.lmatrix('source_smp')

	logger.info('Building RNN encoder-decoder')
	# Encoder 
	embedder = LookupTable(
		length=len(vocab_src), 
		dim=config['embed_src'], 
		weights_init=IsotropicGaussian(0.01),
		biases_init=Constant(0))
	embedding = embedder.apply(source_sentence)
	encoder = Bidirectional(
		GatedRecurrent(
			dim=config['hidden_src'], 
			activation=Tanh(), 
			gate_activation=Tanh(),
			weights_init=Orthogonal(),
			biases_init=Constant(0)))
	encoded = encoder.apply(embedding, embedding)

	### Decoder 
	transition = GatedRecurrent(
		dim=config['hidden_tgt'], 
		name='decoder', 
		weights_init=Orthogonal(),
		biases_init=Constant(0))

	# defult activation in Tanh
	attention = SequenceContentAttention( 
		state_names=transition.apply.states,
		attended_dim=config['hidden_src']*2,
		match_dim=config['hidden_tgt'], 
		name="attention")

	readout = Readout(
		source_names=['states', 
			'feedback', 
			attention.take_glimpses.outputs[0]],
		readout_dim=len(vocab_tgt),
		emitter = SoftmaxEmitter(
			initial_output=0,
			name='emitter'), 
		feedback_brick = LookupFeedback(
			num_outputs=len(vocab_tgt), 
			feedback_dim=config['embed_tgt'], 
			name='feedback'), 
		post_merge=InitializableFeedforwardSequence([
			Bias(dim=config['hidden_tgt'], 
				name='maxout_bias').apply,
			Maxout(num_pieces=2, 
				name='maxout').apply,
			Linear(input_dim=config['hidden_tgt'] / 2, 
				output_dim=config['embed_tgt'],
				use_bias=False, 
				name='softmax0').apply,
			Linear(input_dim=config['embed_tgt'], 
				name='softmax1').apply]),
		merged_dim=config['hidden_tgt'])

	sg = SequenceGenerator(
		readout=readout, 
		transition=transition, 
		attention=attention, 
		weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
			name="generator",
		fork=Fork(
			[name for name in transition.apply.sequences if name != 'mask'], 
			prototype=Linear()),
		add_contexts=True)

	#printchildren(sg, 1)
	#printchildren(encoder, 1)

	generated = sg.generate(
		n_steps=2*source_sentence_smp.shape[1], 
		batch_size=source_sentence_smp.shape[0], 
		attended=encoded, 
		attended_mask=tensor.ones(source_sentence_smp.shape).T)

	cost = sg.cost(
		mask = target_sentence_mask.T, 
		outputs = target_sentence.T, 
		attended = encoded, 
		attended_mask = source_sentence_mask)

	logger.info('Creating computational graph')
	cg = ComputationGraph(cost)

	print cg

	# Initialize model
	logger.info('Initializing model')
	embedder.initialize()
	encoder.initialize()
	sg.initialize()

	# apply dropout for regularization
	if config['dropout'] < 1.0: # dropout is applied to the output of maxout in ghog
		logger.info('Applying dropout')
		dropout_inputs = [x for x in cg.intermediary_variables if x.name == 'maxout_apply_output']
		cg = apply_dropout(cg, dropout_inputs, config['dropout'])

	######## 
	# Print shapes
	shapes = [param.get_value().shape for param in cg.parameters]
	logger.info("Parameter shapes: ")
	for shape, count in Counter(shapes).most_common():
		logger.info('	{:15}: {}'.format(shape, count))
	logger.info("Total number of parameters: {}".format(len(shapes)))

	# Print parameter names
	enc_dec_param_dict = merge(Selector(embedder).get_parameters(), Selector(encoder).get_parameters(), Selector(sg).get_parameters())
	logger.info("Parameter names: ")
	for name, value in enc_dec_param_dict.items():
		logger.info('	{:15}: {}'.format(value.get_value().shape, name))
	logger.info("Total number of parameters: {}".format(len(enc_dec_param_dict)))
	##########

	# Set up training model
	logger.info("Building model")
	training_model = Model(cost)

	# Set extensions
	logger.info("Initializing extensions")
	extensions = [
		FinishAfter(after_n_batches=config['finish_after']),
		TrainingDataMonitoring([cost], after_batch=True),
		Printing(after_batch=True),
		Checkpoint(path=config['saveto'],every_n_batches=config['save_freq'])]

	if config['reload']: 
		extensions.append(Load(path=cofig['saveto'], 
			load_iteration_state=False, 
			load_log=False))

	# Sampling from train and dev 
	'''
	logger.info("Building sampling model")
	embedding_smp = embedder.apply(source_sentence_smp)
	encoded_smp = encoder.apply(embedding_smp, embedding_smp)
	generated_smp = sg.generate(
		n_steps=2*source_sentence_smp.shape[1], 
		batch_size=source_sentence_smp.shape[0], 
		attended=encoded_smp, 
		attended_mask=tensor.ones(source_sentence_smp.shape).T)
	search_model = Model(generated)
	search_model_smp = Model(generated_smp)
	_, samples_tra = VariableFilter(bricks=[sg], name="outputs")(ComputationGraph(generated[1])) 
	_, samples_dev = VariableFilter(bricks=[sg], name="outputs")(ComputationGraph(generated_smp[1])) 
	'''

	# Set up training algorithm
	logger.info("Initializing training algorithm")
	algorithm = GradientDescent(cost=cost, 
		parameters=cg.parameters,
		step_rule=CompositeRule([StepClipping(config['step_clipping']), 
			eval(config['step_rule'])()])
    )
	


	# Training data 
	train_stream = get_src_tgt_stream(config, 
		[config['train_src'],], [config['train_tgt'],], 
		vocab_src, vocab_tgt)
	'''for x in train_stream.get_epoch_iterator(): 
		print x'''

	dev_stream = get_src_tgt_stream(config,
		[config['dev_src'],], [config['dev_tgt'],], 
		vocab_src, vocab_tgt)
	test_stream = get_src_stream(config,
		[config['test_src'],], vocab_src)

	# Initialize main loop
	logger.info("Initializing main loop")
	main_loop = MainLoop(
		model=training_model,
		algorithm=algorithm,
		data_stream=train_stream,
		extensions=extensions)
	main_loop.run()


if __name__ == '__main__':
	config = configure.exe()
	main(config)
