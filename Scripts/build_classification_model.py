import string
import os
import json
from sklearn import svm
from gensim.models import Word2Vec
from DataLoaders import DataLoaderNoTags, DataLoaderWithTags
import numpy as np
import pandas as pd
import pickle

core_dir = os.getcwd()


# based off code from https://www.kaggle.com/code/mehmetlaudatekman/tutorial-word-embeddings-with-svm, will probably need additional reworking for completion
class Sequencer():
    def __init__(self, all_words, seq_len, embedding_matrix):
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        self.vocab = list(set(all_words))
        # temp_vocab = list(set(all_words))
        '''self.vocab = []
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
            self.vocab.append(temp_vocab[ind])'''
        
    
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

# Builds sequencer, based on with or without tags, and whether or not it uses the full dataset
def build_sequencer(use_full_dataset, use_tags):

    # Used to create the paths for loading the approriate word2vec model and dataset, and later writing the output, based on the inputs
    dataset_location = ""
    dataset_model_string = ""
    tags_string = ""
    if use_full_dataset == True:
        dataset_location = "/Datasets/news_set_financial_preprocessed/"
        dataset_model_string = "full_dataset_"
    else:
        dataset_location = "/Datasets/news_set_financial_sampled/"
        dataset_model_string = "sampled_dataset_"
    if use_tags == True:
        tags_string = "w_tags"
    else:
        tags_string = "no_tags"
    
    print("Dataset Location: " + core_dir + dataset_location)
    print("Model: " + core_dir + "/Models/" + dataset_model_string + tags_string + "_word2vec.model")
    dataset_directory = core_dir + dataset_location
    model_directory = core_dir + "/Models/" + dataset_model_string + tags_string + "_word2vec.model"

    w2vmodel = Word2Vec.load(model_directory)
    fixed_dl = []
    dl = ''
    if use_tags == True:
        dl = list(DataLoaderWithTags(dataset_directory))
        print("Tags will be used!")
    else:
        dl = list(DataLoaderNoTags(dataset_directory))
        print("Tags will not be used!")
    for sen in dl:
        for word in sen:
            fixed_dl.append(word)    
    print("This step is now complete!")

    sq = Sequencer(all_words=fixed_dl, seq_len=50, embedding_matrix=w2vmodel.wv)
    pickle.dump(sq, open(core_dir + "/Models/sequencer_" + dataset_model_string + tags_string + ".pkl", "wb"))
    print("Sequencer built and written to file!")

# Arguments are the same as for build_sequencer, and function more or less identically
def build_dataframe(use_full_dataset, use_tags):
    dataset_dir = ""
    sq_name = core_dir + "/Models/sequencer_"
    if use_full_dataset == True:
        dataset_dir = core_dir + "/Datasets/news_set_financial_preprocessed"
        sq_name += "full_dataset_"
    else:
        dataset_dir = core_dir + "/Datasets/news_set_financial_sampled"
        sq_name += "sampled_dataset_"
    if use_tags == True:
        sq_name += "w_tags.pkl"
    else:
        sq_name += "no_tags.pkl"
    
    print("Dataset Directory: " + dataset_dir)
    print("Sequencer Directory: " + sq_name)
    sq = pickle.load(open(sq_name, "rb"))
    print("Sequencer successfully loaded!")

    # Note: will remain commented out until I finalise a dataframe structure
    '''uuid_col = []
    author_col = []
    title_col = []
    text_col = []
    for dirpath, subdirs, files in os.walk(files_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            uuid_col.append(file_json["uuid"])
            author_col.append(file_json["author"])
            fixed_title = [sq.textToVector(fix_text_no_tags(sen)) for sen in file_json["title"]]
            title_col.append(fixed_title)
            fixed_text = [sq.textToVector(fix_text_no_tags(sen)) for sen in file_json["text"]]
            text_col.append(fixed_text)

    df_ind_cols = pd.DataFrame({'UUID': uuid_col, 'Author':author_col, 'Title': title_col, 'Text': text_col})

    pickle.dump(df_ind_cols, open(core_dir + "/Models/ind_cols_df_sampled_no_tags.pkl", "wb"))
    print("We have made it this far.")'''

def main():
    # build_sequencer(True, False)
    # build_sequencer(True, True)
    # build_sequencer(False, False)
    # build_sequencer(False, True)
    build_dataframe(True, False)
    # build_dataframe(True, True)
    # build_dataframe(False, False)
    # build_dataframe(False, True)

main()
