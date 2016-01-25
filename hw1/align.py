import numpy as np
import cPickle as pkl

def prep():
	BITEXT = 'data/dev-test-train.de-en'
	NUM_SENTS = 100000

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
	for d in set(lt): 
		eng_word_dict[d]=lt.count(d)
	print 'Sorting English...'
	eng_word_list = sorted(eng_word_dict.keys(), key=eng_word_dict.get, reverse=True)
	for i in range(len(eng_word_list)):
		eng_word_dict[eng_word_list[i]]=i
	eng_corp = [np.array([eng_word_dict[w] for w in li]) for li in eng_corp]

	print 'Seperating French...'
	lt = [item for sublist in fra_corp for item in sublist]
	print 'Counting French...'
	fra_word_dict = {}
	for d in set(lt): 
		fra_word_dict[d]=lt.count(d)
	print 'Sorting French...'
	fra_word_list = sorted(fra_word_dict.keys(), key=fra_word_dict.get, reverse=True)
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
	t_eng_given_fra = np.ones((fra_len,eng_len))/eng_len
	with open('corp.pkl', 'rb') as f:
		fra_corp = pkl.load(f)
		eng_corp = pkl.load(f)
	ppx = np.zeros((len(fra_corp),))

	for itr in range(100):
		# E-STEP 
		count = np.zeros_like(t_eng_given_fra)
		change = False
		for i in range(NUM_SENTS): 
			s_t_eng_given_fra = t_eng_given_fra[np.ix_(fra_corp[i], eng_corp[i])]
			s_t_eng_given_fra = s_t_eng_given_fra/s_t_eng_given_fra.sum(axis=1)[:,np.newaxis]
			count[np.ix_(fra_corp[i], eng_corp[i])] = count[np.ix_(fra_corp[i], eng_corp[i])]+s_t_eng_given_fra
			p = np.log2(s_t_eng_given_fra.sum(axis=0)).sum()
			ppx[i] = len(eng_corp[i])*np.log2(1.0/len(fra_corp[i]))+p
			
		# M-STEP
		t_eng_given_fra = count/count.sum(axis=1)[:,np.newaxis]
		for i in range(NUM_SENTS): 
			s_t_eng_given_fra = t_eng_given_fra[np.ix_(fra_corp[i], eng_corp[i])]
			ppx[i] = len(eng_corp[i])*np.log2(1.0/len(fra_corp[i]))+np.log2(s_t_eng_given_fra.sum(axis=0)).sum()
		print "log2ppx:", -ppx.sum()
		#print t_eng_given_fra
		#print t_eng_given_fra.sum(axis=1)[0]
		#heatmap = plt.pcolor(t_eng_given_fra, cmap = cm.Greys_r)
		#plt.gca().set_aspect('equal')
		#plt.show()


	for i in range(300):
		s_t_eng_given_fra = t_eng_given_fra[np.ix_(fra_corp[i], eng_corp[i])]
		aln = s_t_eng_given_fra.argmax(axis=1)
		#print len(aln), len(fra_corp[i])
		print ' '.join([str(a)+'-'+str(aln[a]) for a in range(len(aln))])
		#print aln



if __name__ == "__main__":
    e, f = prep()
    em(e,f)