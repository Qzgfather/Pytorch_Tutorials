import pickle as pkl
word_to_id = pkl.load(open('./THUCNews/data/vocab.pkl', 'rb'))
print(word_to_id)
