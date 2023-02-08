import string
import os
import json
from sklearn import svm
from gensim.models import Word2Vec
from DataLoaders import DataLoaderNoTags
import numpy as np
import pickle

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
            count = len([0 for w in all_words if w == word]) # the single longest part of the process, seemingly
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
        len_v = len(text)-1 if len(text) < self.seq_len else self.seq_len-1
        vec = []
        for tok in text[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                pass
        
        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(100,))
        
        return np.asarray(vec).flatten()

def fix_text_no_tags(input_text):
    output = []
    for sen in input_text:
        fixed_sen = []
        for word in sen:
            fixed_sen.append(word[0])
        output.append(fixed_sen)
    return output

def fix_text_w_tags(input_text):
    output = []
    for sen in input_text:
        fixed_sen = []
        for word in sen:
            new_word = word[0]+word[1]
            fixed_sen.append(new_word)
        output.append(fixed_sen)
    return output

def main():
    w2vmodel = Word2Vec.load(core_dir + "/Models/sampled_dataset_no_tags.model")
    files_dir = core_dir + "/Datasets/news_set_financial_sampled/"

    # TODO: consider precomputing this, and writing to file serialised
    '''dl = list(DataLoaderNoTags(files_dir))
    fixed_dl = []
    for sen in dl:
        for word in sen:
            fixed_dl.append(word)'''

    fptr = open(core_dir+"/Datasets/test_sample_article.json")
    file_json = json.load(fptr)
    fptr.close()
    file_title = fix_text_no_tags(file_json["title"])
    file_text = fix_text_no_tags(file_json["text"])
    file_content = []
    for sen in file_title:
        for word in sen:
            file_content.append(word)
    for sen in file_text:
        for word in sen:
            file_content.append(word)
    
    
    sq = Sequencer(all_words=file_content, max_words=100, seq_len=50, embedding_matrix=w2vmodel.wv)
    # pickle.dump(sq, open(core_dir + "/Models/sq_sampled.pkl", 'wb')) //commented out because most recent testing didn't use it
    # sq = pickle.load(open(core_dir+"/Models/sq_sampled.pkl", "rb"))

    sequenced_title = [sq.textToVector(sen) for sen in file_title]

    sequenced_text = [sq.textToVector(sen) for sen in file_text]

    # Commented out until I can find a way of doing this process quickly
    '''uuid_col = []
    title_col = []
    text_col = []
    for dirpath, subdirs, files in os.walk(files_dir):
        for f in files:
            file_path = os.join(dirpath, f)
            fptr = open(f_name)
            file_json = json_load(fptr)
            fptr.close()
            uuid_col.append(file_json["uuid"])
            fixed_title = [sq.textToVector(sen) for sen in file_json["title"]]
            title_col.append(fixed_title)
            fixed_text = [sq.textToVector(sen) for sen in file_json["text"]]'''
    print("We have made it this far.")

main()
