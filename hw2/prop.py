with open('glove300dact3.txt', 'r') as fl: 
    for ln in fl: 
        print ' '.join(ln.split()[1:])