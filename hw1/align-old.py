import numpy as np
from scipy.sparse import coo_matrix
import cPickle as pkl
NUM_SENTS = 100000

def prep():
	BITEXT = 'data/dev-test-train.de-en'
	
	# eng_corp, fra_corp
	# eng_word_dict, fra_word_dict
	# eng_word_list, fra_word_list
	
	eng_corp = []
	fra_corp = []
	print 'Reading lines...'
	with open(BITEXT,'r') as fi:
		for i in range(NUM_SENTS):
			pair = fi.readline().decode('utf8').lower().split('|||')
			eng_corp.append(pair[1].split())
			fra_corp.append(pair[0].split())

	print 'Seperating English...'
	lt = [item for sublist in eng_corp for item in sublist]
	print 'Counting English...'
	eng_word_dict = {}
	eng_word_list = list(set(lt))
	print 'Making dict... '
	for i in range(len(eng_word_list)):
		eng_word_dict[eng_word_list[i]]=i
	print 'Making array'
	eng_corp = [np.array([eng_word_dict[w] for w in li]) for li in eng_corp]

	print 'Seperating French...'
	lt = [item for sublist in fra_corp for item in sublist]
	print 'Counting French...'
	fra_word_dict = {}
	fra_word_list = list(set(lt))
	for i in range(len(fra_word_list)):
		fra_word_dict[fra_word_list[i]]=i
	fra_corp = [np.array([fra_word_dict[w] for w in li]) for li in fra_corp]

	#from matplotlib import pyplot as plt
	#import matplotlib.cm as cm
	print 'Pickling...'
	with open('corp.pkl', 'wb') as f:
		pkl.dump(fra_corp, f, pkl.HIGHEST_PROTOCOL)
		pkl.dump(eng_corp, f, pkl.HIGHEST_PROTOCOL)
	
	### ENGLISH GIVEN FRANCH 
	print 'len(fra_word_list),len(eng_word_list)', len(fra_word_list),len(eng_word_list)
	return len(fra_word_list),len(eng_word_list)

def em(fra_len, eng_len):
	# t_eng_given_fra, temp: count, size: fra x eng
	# INITIALIZATION 
	t_eng_given_fra = np.ones((fra_len,eng_len))#/eng_len
	with open('corp.pkl', 'rb') as f:
		fra_corp = pkl.load(f)
		eng_corp = pkl.load(f)
	ppx = np.zeros((NUM_SENTS,))
	print 'E-STEP'
	for itr in range(600000):
		# E-STEP 
		#count = coo_matrix(np.zeros_like(t_eng_given_fra))
		count_row = np.ndarray(0,)
		count_col = np.ndarray(0,)
		count_data = np.ndarray(0,)
		s_t_eng_given_fra = t_eng_given_fra[np.ix_(fra_corp[itr], eng_corp[itr])]
		np.true_divide(s_t_eng_given_fra, s_t_eng_given_fra.sum(axis=1)[:,np.newaxis], s_t_eng_given_fra)
		c = coo_matrix(s_t_eng_given_fra)
		count_row = np.append(count_row, fra_corp[itr][c.row])
		count_col = np.append(count_col, eng_corp[itr][c.col])
		count_data = np.append(count_data, c.data)
		#count[np.ix_(fra_corp[itr], eng_corp[itr])] = count[np.ix_(fra_corp[itr], eng_corp[itr])]+s_t_eng_given_fra
		ppx[itr] = len(eng_corp[itr])*np.log2(1.0/len(fra_corp[itr]))+np.log2(s_t_eng_given_fra.sum(axis=0)).sum()
		if itr%1000 is 0:
			print itr
		if itr%100000 == 99999:
			print itr
			# M-STEP
			count = coo_matrix((count_data,(count_row,count_col)), shape=t_eng_given_fra.shape)
			#print count.shape, count.sum(axis=1).shape, count.sum(axis=1)[:,np.newaxis].shape, t_eng_given_fra.shape
			#np.true_divide(count, count.sum(axis=1)[:,np.newaxis], t_eng_given_fra)
			print 'dividing...'
			np.true_divide(count, count.sum(axis=1), t_eng_given_fra) # Because count is coo, after sum dim is different 
			print t_eng_given_fra
			print "log2ppx:", -ppx.sum()#, itr+1
			with open('output.txt', 'w') as f:
				for i in range(NUM_SENTS):
					s_t_eng_given_fra = t_eng_given_fra[np.ix_(fra_corp[i], eng_corp[i])]
					aln = s_t_eng_given_fra.argmax(axis=1)
					f.write(' '.join([str(a)+'-'+str(aln[a]) for a in range(len(aln))]))
			with open('t_eng_given_fra'+str(i), 'wb') as f:
				t_eng_given_fra.dump(f)
	#heatmap = plt.pcolor(t_eng_given_fra, cmap = cm.Greys_r)
	#plt.gca().set_aspect('equal')
	#plt.show()

if __name__ == "__main__":
	#e, f = prep()
	em(81394, 34503)