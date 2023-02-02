import string
import os
import json
from sklearn import svm
from gensim.models import Word2Vec
import numpy as np

core_dir = os.getcwd()


# based off code from https://www.kaggle.com/code/mehmetlaudatekman/tutorial-word-embeddings-with-svm, will probably need additional reworking for completion
class Sequencer():
    def __init__(self, all_words, max_words, seq_len, embedding_matrix):
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}

        for word in temp_vocab:
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))
        
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1],counts[i] = counts[i],counts[i+1]
                    indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                else:
                    cnt += 1
        
        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])
        
    
    def textToVector(self,text):
        # NOTE: will need to rework to account for my format of word and POS tag
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass
        
        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(100,))
        
        return np.asarray(vec).flatten()

def main():
    w2vmodel = Word2Vec.load(core_dir + "/Models/full_dataset_no_tags_word2vec.model")
    fptr = open(core_dir + "/Datasets/test_sample_article.json")
    file_json = json.load(fptr)
    json_text = file_json["text"]
    fixed_text = []
    for sen in json_text:
        for word in sen:
            fixed_text.append(word[0])
    fptr.close()

    sq = Sequencer(all_words=fixed_text, max_words=1200, seq_len=1000, embedding_matrix=w2vmodel.wv )


main()
