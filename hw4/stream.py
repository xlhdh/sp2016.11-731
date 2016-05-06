from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
# Adapted from: 
# https://github.com/mila-udem/blocks-examples/blob/master/machine_translation/stream.py

def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])

class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])

def get_test_stream(sfiles, svocab_dict): 
	dataset = TextFile(sfiles, svocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')
	stream = Merge([dataset.get_example_stream(),], ('source', ))
	stream = Batch(
        stream, iteration_scheme=ConstantScheme(10))
	stream = Padding(stream)
	return stream

def get_train_stream(configuration, sfiles, tfiles, svocab_dict, tvocab_dict):

	s_dataset = TextFile(sfiles, svocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')
	t_dataset = TextFile(tfiles, tvocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')

	# Merge 
	stream = Merge([s_dataset.get_example_stream(),
                    t_dataset.get_example_stream()],
                   ('source', 'target'))
	# Filter -- TODO 
	stream = Filter(stream, predicate=_too_long(seq_len=configuration['seq_len']))

	# Map - no need 

	# Batch - Sort 
	stream = Batch(stream, 
		iteration_scheme=ConstantScheme(
			configuration['batch_size']*configuration['sort_k_batches']))
	stream = Mapping(stream, SortMapping(_length))
	stream = Unpack(stream)
	stream = Batch(
        stream, iteration_scheme=ConstantScheme(configuration['batch_size']))

	# Pad 
	# Note that </s>=0. Fuel only allows padding 0 by default 
	masked_stream = Padding(stream)

	return masked_stream

def get_dev_stream(sfiles, tfiles, svocab_dict, tvocab_dict):

	s_dataset = TextFile(sfiles, svocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')
	t_dataset = TextFile(tfiles, tvocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')

	# Merge 
	stream = Merge([s_dataset.get_example_stream(),
                    t_dataset.get_example_stream()],
                   ('source', 'target'))
	# Batch - Sort 
	stream = Batch(stream, 
		iteration_scheme=ConstantScheme(1006))
	# Pad 
	# Note that </s>=0. Fuel only allows padding 0 by default 
	masked_stream = Padding(stream)

	return masked_stream

