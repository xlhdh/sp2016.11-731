def exe(): 
	config = {}
	config['datadir'] = 'data/'

	config['train_src'] = config['datadir'] + 'train.src'
	config['dev_src'] = config['datadir'] + 'dev.src'
	config['test_src'] = config['datadir'] + 'test.src'

	config['train_tgt'] = config['datadir'] + 'train.tgt'
	config['dev_tgt'] = config['datadir'] + 'dev.tgt'

	# MAX Length: src = 61, tgt = 67
	config['seq_len'] = 20

	config['embed_src'] = 16
	config['embed_tgt'] = 16

	config['hidden_src'] = 16
	config['hidden_tgt'] = 16

	config['batch_size'] = 80
	config['sort_k_batches'] = 40

	config['dropout'] = 0.9

	config['saveto'] = 'try' \
		+str(config['embed_src'])+'_' \
		+str(config['embed_tgt'])+'_' \
		+str(config['hidden_src'])+'_' \
		+str(config['hidden_tgt'])

	config['reload'] = False

	config['finish_after'] = 1000000
	config['save_freq'] = 50

	config['step_rule'] = 'Adam'
	config['step_clipping'] = 1.


	return config 