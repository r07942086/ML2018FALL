
#!/bin/sh
wget -O word2vec_76165.model.trainables.syn1neg.npy 'https://www.dropbox.com/s/sudrb3owue72zix/word2vec_76165.model.trainables.syn1neg.npy?dl=1'
wget -O word2vec_76165.model.wv.vectors.npy 'https://www.dropbox.com/s/ghxn5qz78lnkomx/word2vec_76165.model.wv.vectors.npy?dl=1'
python3 hw4_76165_pre.py ${1} ${2} ${3} 
