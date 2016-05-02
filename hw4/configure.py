def exe(): 
	config = {}
	config['datadir'] = 'data/'

	config['train_src'] = config['datadir'] + 'train.src'
	config['dev_src'] = config['datadir'] + 'dev.src'
	config['test_src'] = config['datadir'] + 'test.src'

	config['train_tgt'] = config['datadir'] + 'train.tgt'
	config['dev_tgt'] = config['datadir'] + 'dev.tgt'

	# MAX Length: src = 61, tgt = 67
	config['seq_len_src'] = 62
	config['seq_len_tgt'] = 68

	config['embed_src'] = 64
	config['embed_tgt'] = 64

	config['hidden_src'] = 128
	config['hidden_tgt'] = 128

	config['batch_size'] = 1024
	config['sort_k_batches'] = 40

	config['dropout'] = 0.9

	config['saveto'] = 'try'+str(config['embed_src'])+'_' \
		+str(config['embed_tgt'])+'_' \
		+str(config['hidden_src'])+'_' \
		+str(config['hidden_tgt'])

	config['reload'] = False

	config['finish_after'] = 1000000
	config['save_freq'] = 50

	config['step_rule'] = 'AdaDelta'
	config['step_clipping'] = 1.


	return config 