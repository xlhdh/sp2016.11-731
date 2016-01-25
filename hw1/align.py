import numpy as np

BITEXT = 'data/dev-test-train.de-en'
NUM_SENTS = 20#0000

def main():
	# eng_corp, fra_corp
	# eng_word_dict, fra_word_dict
	# eng_word_list, fra_word_list
	# t_eng_given_fra, temp: count, size: eng x fra
	# 

	eng_corp = []
	fra_corp = []
	with open(BITEXT,'r') as fi:
		for i in range(NUM_SENTS):
			pair = fi.readline().decode('utf8').lower().split('|||')
			eng_corp.append(pair[1].split())
			fra_corp.append(pair[0].split())

	lt = [item for sublist in eng_corp for item in sublist]
	eng_word_dict = {}
	for d in set(lt): 
		eng_word_dict[d]=lt.count(d)
	eng_word_list = sorted(eng_word_dict.keys(), key=eng_word_dict.get, reverse=True)
	for i in range(len(eng_word_list)):
		eng_word_dict[eng_word_list[i]]=i
	eng_corp = [np.array([eng_word_dict[w] for w in li]) for li in eng_corp]

	lt = [item for sublist in fra_corp for item in sublist]
	fra_word_dict = {}
	for d in set(lt): 
		fra_word_dict[d]=lt.count(d)
	fra_word_list = sorted(fra_word_dict.keys(), key=fra_word_dict.get, reverse=True)
	for i in range(len(fra_word_list)):
		fra_word_dict[fra_word_list[i]]=i
	fra_corp = [np.array([fra_word_dict[w] for w in li]) for li in fra_corp]

	
	### ENGLISH GIVEN FRANCH 
	print (len(fra_word_list),len(eng_word_list))

	# INITIALIZATION 
	t_eng_given_fra = np.ones((len(fra_word_list),len(eng_word_list)))
	t_eng_given_fra = t_eng_given_fra/t_eng_given_fra.sum(axis=0)
	
	# E-STEP 
	count = np.zeros_like(t_eng_given_fra)
	for i in range(NUM_SENTS): 
		s_t_eng_given_fra = t_eng_given_fra[np.ix_(fra_corp[i], eng_corp[i])]
		count[np.ix_(fra_corp[i], eng_corp[i])] = count[np.ix_(fra_corp[i], eng_corp[i])]+s_t_eng_given_fra
	print count




if __name__ == "__main__":
    main()