import configure
from collections import Counter
from toolz import merge

from util import text_to_dict, InitializableFeedforwardSequence, Sampler, Plotter
from stream import get_train_stream, get_dev_stream, get_test_stream

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
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.saveload import Load, Checkpoint
try:
	from blocks_extras.extensions.plot import Plot
	BOKEH_AVAILABLE = True
except ImportError:
	BOKEH_AVAILABLE = False
BOKEH_AVAILABLE = False

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, Adam, CompositeRule)

from blocks.monitoring.aggregation import TakeLast, Mean

import theano
theano.config.compute_test_value = 'warn'
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def printchildren(parent, i):
	print '<....>'*i, parent.name
	if parent.initialized: 
		for p in parent.parameters: 
			print '<....>'*i + '<>', p, p.get_value().shape
	for child in parent.children: 
		printchildren(child, i+1)


def main(config): 
	vocab_src, _ = text_to_dict([config['train_src'],
		config['dev_src'], config['test_src']])
	vocab_tgt, cabvo = text_to_dict([config['train_tgt'],
		config['dev_tgt']])

	# Create Theano variables
	logger.info('Creating theano variables')
	source_sentence = tensor.lmatrix('source')
	source_sentence_mask = tensor.matrix('source_mask')
	target_sentence = tensor.lmatrix('target')
	target_sentence_mask = tensor.matrix('target_mask')
	source_sentence.tag.test_value = [[13, 20, 0, 20, 0, 20, 0],
										[1, 4, 8, 4, 8, 4, 8],]
	source_sentence_mask.tag.test_value = [[0, 1, 0, 1, 0, 1, 0],
											[1, 0, 1, 0, 1, 0, 1],]
	target_sentence.tag.test_value = [[0,1,1,5],
										[2,0,1,0],]
	target_sentence_mask.tag.test_value = [[0,1,1,0],
											[1,1,1,0],]


	logger.info('Building RNN encoder-decoder')
	### Building Encoder 
	embedder = LookupTable(
		length=len(vocab_src), 
		dim=config['embed_src'], 
		weights_init=IsotropicGaussian(),
		biases_init=Constant(0.0), 
		name='embedder')
	transformer = Linear(
		config['embed_src'], 
		config['hidden_src']*4, 
		weights_init=IsotropicGaussian(),
		biases_init=Constant(0.0), 
		name='transformer')
	encoder = Bidirectional(
		LSTM(
			dim=config['hidden_src'], 
			weights_init=IsotropicGaussian(0.01),
			biases_init=Constant(0.0)),
		name='encoderBiLSTM'
		)
	encoder.prototype.weights_init = Orthogonal()
	
	### Building Decoder 
	transition = LSTM(
		dim=config['hidden_tgt'], 
		weights_init=IsotropicGaussian(0.01),
		biases_init=Constant(0.0), 
		name='decoderGRU')

	attention = SequenceContentAttention( 
		state_names=transition.apply.states[:1], # default activation is Tanh
		attended_dim=config['hidden_src']*2,
		match_dim=config['hidden_tgt'], 
		name="attention")

	readout = Readout(
		source_names=['states', 
			'feedback', 
			attention.take_glimpses.outputs[0]],
		readout_dim=len(vocab_tgt),
		emitter = SoftmaxEmitter(
			name='emitter'), 
		feedback_brick = LookupFeedback(
			num_outputs=len(vocab_tgt), 
			feedback_dim=config['embed_tgt'], 
			name='feedback'), 
		post_merge=InitializableFeedforwardSequence([
			Bias(dim=config['hidden_tgt'], 
				name='softmax_bias').apply,
			Linear(input_dim=config['hidden_tgt'], 
				output_dim=config['embed_tgt'],
				use_bias=False, 
				name='softmax0').apply,
			Linear(input_dim=config['embed_tgt'], 
				name='softmax1').apply]),
		merged_dim=config['hidden_tgt'])

	decoder = SequenceGenerator(
		readout=readout, 
		transition=transition, 
		attention=attention, 
		weights_init=IsotropicGaussian(0.01), 
		biases_init=Constant(0),
		name="generator",
		fork=Fork(
			[name for name in transition.apply.sequences if name != 'mask'], 
			prototype=Linear()),
		add_contexts=True)
	decoder.transition.weights_init = Orthogonal()

	#printchildren(encoder, 1)
	# Initialize model
	logger.info('Initializing model')
	embedder.initialize()
	transformer.initialize()
	encoder.initialize()
	decoder.initialize()
	
	# Apply model 
	embedded = embedder.apply(source_sentence)
	tansformed1 = transformer.apply(embedded)
	encoded = encoder.apply(tansformed1)[0]
	generated = decoder.generate(
		n_steps=2*source_sentence.shape[1], 
		batch_size=source_sentence.shape[0], 
		attended = encoded.dimshuffle(1,0,2), 
		attended_mask=tensor.ones(source_sentence.shape).T
		)
	print 'Generated: ', generated
	samples = generated[1]
	samples.name = 'samples'
	samples_cost = generated[4]
	samples_cost = 'sampling_cost'
	cost = decoder.cost(
		mask = target_sentence_mask.T, 
		outputs = target_sentence.T, 
		attended = encoded.dimshuffle(1,0,2), 
		attended_mask = source_sentence_mask.T)
	cost.name = 'target_cost'
	cost.tag.aggregation_scheme = TakeLast(cost)
	model = Model(cost)
	
	logger.info('Creating computational graph')
	cg = ComputationGraph(cost)
	
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

	printchildren(embedder, 1)
	printchildren(transformer, 1)
	printchildren(encoder, 1)
	printchildren(decoder, 1)
	# Print parameter names
	# enc_dec_param_dict = merge(Selector(embedder).get_parameters(), Selector(encoder).get_parameters(), Selector(decoder).get_parameters())
	# enc_dec_param_dict = merge(Selector(decoder).get_parameters())
	# logger.info("Parameter names: ")
	# for name, value in enc_dec_param_dict.items():
	# 	logger.info('	{:15}: {}'.format(value.get_value().shape, name))
	# logger.info("Total number of parameters: {}".format(len(enc_dec_param_dict)))
	##########

	# Training data 
	train_stream = get_train_stream(config, 
		[config['train_src'],], [config['train_tgt'],], 
		vocab_src, vocab_tgt)
	dev_stream = get_dev_stream(
		[config['dev_src'],], [config['dev_tgt'],], 
		vocab_src, vocab_tgt)
	test_stream = get_test_stream([config['test_src'],], vocab_src)

	# Set extensions
	logger.info("Initializing extensions")
	extensions = [
		FinishAfter(after_n_batches=config['finish_after']),
		TrainingDataMonitoring([cost], 
			prefix="tra", 
			after_batch=True),
		DataStreamMonitoring(variables=[cost], 
			data_stream=dev_stream, 
			prefix="dev", 
			after_batch=True), 
		Sampler(
			model=Model(samples), 
			data_stream=dev_stream,
			vocab=cabvo,
			saveto=config['saveto']+'dev',
			every_n_batches=config['save_freq']), 
		Sampler(
			model=Model(samples), 
			data_stream=test_stream,
			vocab=cabvo,
			saveto=config['saveto']+'test',
			on_resumption=True,
			before_training=True), 
		Plotter(saveto=config['saveto'], after_batch=True),
		Printing(after_batch=True),
		Checkpoint(
			path=config['saveto'], 
			parameters = cg.parameters,
			save_main_loop=False,
			every_n_batches=config['save_freq'])]
	if BOKEH_AVAILABLE: 
		Plot('Training cost', channels=[['target_cost']], after_batch=True)
	if config['reload']: 
		extensions.append(Load(path=config['saveto'], 
			load_iteration_state=False, 
			load_log=False))
	else: 
		with open(config['saveto']+'.txt', 'w') as f: 
			pass 

	# Set up training algorithm
	logger.info("Initializing training algorithm")
	algorithm = GradientDescent(cost=cost, 
		parameters=cg.parameters,
		step_rule=CompositeRule([StepClipping(config['step_clipping']), 
			eval(config['step_rule'])()])
    )

	# Initialize main loop
	logger.info("Initializing main loop")
	main_loop = MainLoop(
		model=model,
		algorithm=algorithm,
		data_stream=train_stream,
		extensions=extensions)
	main_loop.run()


if __name__ == '__main__':
	config = configure.exe()
	main(config)
