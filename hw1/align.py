import numpy as np
from scipy.sparse import coo_matrix, dia_matrix, csr_matrix, issparse
import cPickle as pkl
NUM_SENTS = 50
BITEXT = 'data/dev-test-train.de-en'

def prep():
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
	with open(str(NUM_SENTS)+'corp.pkl', 'wb') as f:
		pkl.dump(fra_corp, f, pkl.HIGHEST_PROTOCOL)
		pkl.dump(eng_corp, f, pkl.HIGHEST_PROTOCOL)
	
	### ENGLISH GIVEN FRANCH 
	print 'len(fra_word_list),len(eng_word_list)', len(fra_word_list),len(eng_word_list)
	return len(fra_word_list),len(eng_word_list)

def em(fra_len, eng_len):
	# t_eng_given_fra, temp: count, size: fra x eng
	# INITIALIZATION 
	t_eng_given_fra = np.random.rand(fra_len,eng_len)#/eng_len
	count_row = []
	count_col = []
	count_data = []
	with open(str(NUM_SENTS)+'corp.pkl', 'rb') as f:
		fra_corp = pkl.load(f)
		eng_corp = pkl.load(f)
	ppx = np.zeros((NUM_SENTS,))
	#from matplotlib import pyplot as plt
	#import matplotlib.cm as cm
	
	for itr in range(8):
		# E-STEP
		outstr = []
		for eitr in range(NUM_SENTS):
			s_t_e_f = t_eng_given_fra[np.ix_(fra_corp[eitr], eng_corp[eitr])]
			s_t_e_f = s_t_e_f.reshape((len(fra_corp[eitr]),len(eng_corp[eitr])))
			if np.any(np.where(s_t_e_f.sum(axis=1)==0)):
				print fra_corp[eitr]
				print eng_corp[eitr]
				print s_t_e_f
				print s_t_e_f.sum(axis=1)==0
			
			#s_norm = np.diagflat(np.reciprocal(s_t_e_f.sum(axis=1)))
			#s_t_e_f = s_norm.dot(s_t_e_f)
			s_norm = np.diagflat(np.reciprocal(s_t_e_f.sum(axis=0)))
			#print s_t_e_f.sum(axis=0)
			s_t_e_f = s_t_e_f.dot(s_norm)
			if eitr < 150:
				aln = np.asarray(s_t_e_f).argmax(axis=1)
				#print s_t_e_f
				#print aln
				outstr.append(' '.join([str(a)+'-'+str(aln[a]) for a in range(len(fra_corp[eitr]))]))
				#exit()
			ppx[eitr] = -len(eng_corp[eitr])*np.log2(len(fra_corp[eitr]))+np.log2(s_t_e_f.sum(axis=0)).sum()
			#ppx[eitr] = np.log2(s_t_e_f.sum(axis=0).prod()/(np.power(len(fra_corp[eitr]),len(eng_corp[eitr]))))
			s_coo=coo_matrix(s_t_e_f)
			count_row.append(fra_corp[eitr][s_coo.row])
			count_col.append(eng_corp[eitr][s_coo.col])
			count_data.append(s_coo.data)
			if eitr%10000==9999:
				print eitr+1
		with open('output'+str((itr+1))+'.txt', 'w') as f:
			f.write('\n'.join(outstr))

		# M-STEP
		print "log2ppx:", -ppx.sum()#, itr+1
		count_row = np.concatenate(count_row)
		count_col = np.concatenate(count_col)
		count_data = np.concatenate(count_data)
		count = coo_matrix((count_data,(count_row,count_col)), shape=t_eng_given_fra.shape)
		print 'Dividing...'
		t_save = dia_matrix((np.reciprocal(count.sum(axis=1).flatten()),[0,]), shape=(fra_len, fra_len)).dot(count).tocsr()
		#plt.pcolor(np.array(t_eng_given_fra))
		#plt.show()
		with open('t_eng_given_fra'+str((itr+1))+'.npy', 'wb') as f:
			np.save(f, t_save)
		t_eng_given_fra = t_save.todense()
		count_row = []
		count_col = []
		count_data = []

if __name__ == "__main__":
	e, f = prep()
	em(e, f)
	#em(81394, 34503)
