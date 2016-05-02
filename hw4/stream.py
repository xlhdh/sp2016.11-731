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

def get_src_stream(configuration, sfiles, svocab_dict): 
	dataset = TextFile(sfiles, svocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')
	stream = DataStream(dataset)
	stream = Batch(
        stream, iteration_scheme=ConstantScheme(configuration['batch_size']))
	stream = Padding(stream)
	return stream

def get_src_tgt_stream(configuration, sfiles, tfiles, svocab_dict, tvocab_dict):

	s_dataset = TextFile(sfiles, svocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')
	t_dataset = TextFile(tfiles, tvocab_dict, bos_token=None, eos_token=None,\
		unk_token='<unk>', level='word', preprocess=None, encoding='utf8')

	# Merge 
	stream = Merge([s_dataset.get_example_stream(),
                    t_dataset.get_example_stream()],
                   ('source', 'target'))
	# Filter -- TODO 

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

