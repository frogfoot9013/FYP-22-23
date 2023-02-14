import string
import os
import json
from sklearn import svm
from gensim.models import Word2Vec
from DataLoaders import DataLoaderNoTags, DataLoaderWithTags
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

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
def build_dataframes(use_full_dataset, use_tags):
    dataset_dir = ""
    sq_name = core_dir + "/Models/sequencer_"
    ind_cols_df_name = core_dir + "/Models/df_ind_cols_"
    dep_cols_df_name = core_dir + "/Models/df_dep_cols_"
    if use_full_dataset == True:
        dataset_dir = core_dir + "/Datasets/news_set_financial_preprocessed/"
        sq_name += "full_dataset_"
        ind_cols_df_name += "full_dataset_"
        dep_cols_df_name += "full_dataset_"
    else:
        dataset_dir = core_dir + "/Datasets/news_set_financial_sampled/"
        sq_name += "sampled_dataset_"
        ind_cols_df_name += "sampled_dataset_"
        dep_cols_df_name += "sampled_dataset_"
    if use_tags == True:
        sq_name += "w_tags.pkl"
        ind_cols_df_name += "w_tags.pkl"
        dep_cols_df_name += "w_tags.pkl"
    else:
        sq_name += "no_tags.pkl"
        ind_cols_df_name += "no_tags.pkl"
        dep_cols_df_name += "no_tags.pkl"
    
    print("Dataset Directory: " + dataset_dir)
    print("Sequencer Directory: " + sq_name)
    sq = pickle.load(open(sq_name, "rb"))
    print("Sequencer successfully loaded!")

    uuid_col = []
    author_col = []
    # site_col = []
    title_col = []
    text_col = []
    name_col = []
    sentiment_col = []
    for dirpath, subdirs, files in os.walk(dataset_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            uuid_col.append(file_json["uuid"])
            author_col.append(file_json["author"])
            # site_col.append(file_json["site"]) # Note: consider one-hot encoding this variable
            fixed_title = [sq.textToVector(fix_text_no_tags(sen)) for sen in file_json["title"]]
            title_col.append(fixed_title)
            fixed_text = [sq.textToVector(fix_text_no_tags(sen)) for sen in file_json["text"]]
            text_col.append(fixed_text)
            file_entities = file_json["entities"]
            # structure sentiment through this format
            doc_names = []
            doc_sentiments = []
            for en in file_entities["persons"]:
                doc_names.append(en["name"])
                doc_sentiments.append(en["sentiment"])
                # doc_sentiments.append(process_sentiment(en["sentiment"]))
            for en in file_entities["locations"]:
                doc_names.append(en["name"])
                doc_sentiments.append(en["sentiment"])
                # doc_sentiments.append(process_sentiment(en["sentiment"]))
            for en in file_entities["organizations"]:
                doc_names.append(en["name"])
                doc_sentiments.append(en["sentiment"])
                # doc_sentiments.append(process_sentiment(en["sentiment"]))
            name_col.append(doc_names)
            sentiment_col.append(doc_sentiments)

    sentiment_binariser = MultiLabelBinarizer()
    sentiment_binariser.fit([["positive", "neutral", "negative", "none"]])
    fixed_sentiment = []
    for el in sentiment_col:
        fixed_sentiment.append(sentiment_binariser.fit_transform(el))
    print("Has this binarisation worked?")
    df_dep_cols = pd.DataFrame({'UUID': uuid_col, 'Name': name_col, 'Sentiment': fixed_sentiment})
    df_ind_cols = pd.DataFrame({'UUID': uuid_col, 'Author': author_col, 'Title': title_col, 'Text': text_col})

    pickle.dump(df_ind_cols, open(ind_cols_df_name, "wb"))
    pickle.dump(df_dep_cols, open(dep_cols_df_name, "wb"))
    print("Columns constructed and written!")

def build_model(use_full_dataset, use_tags):
    ind_cols_df_name = core_dir + "/Models/df_ind_cols_"
    dep_cols_df_name = core_dir + "/Models/df_dep_cols_"
    if use_full_dataset == True:
        ind_cols_df_name += "full_dataset_"
        dep_cols_df_name += "full_dataset_"
    else:
        ind_cols_df_name += "sampled_dataset_"
        dep_cols_df_name += "sampled_dataset_"
    if use_tags == True:
        ind_cols_df_name += "w_tags.pkl"
        dep_cols_df_name += "w_tags.pkl"
    else:
        ind_cols_df_name += "no_tags.pkl"
        dep_cols_df_name += "no_tags.pkl"
    print(ind_cols_df_name)
    print(dep_cols_df_name)
    
    df_ind_cols = pickle.load(open(ind_cols_df_name, "rb"))
    df_dep_cols = pickle.load(open(dep_cols_df_name, "rb"))
    print("Columns loaded!")
    dep_cols_names = 'Sentiment'
    ind_cols_names= ['Title','Text']
    ind_cols_loc = df_ind_cols.loc[:,ind_cols_names]
    dep_cols_loc = df_dep_cols.loc[:,dep_cols_names]
    print("Are we at least getting this far?")
    
    multi_output = MultiOutputClassifier(estimator=SVC())
    crv = cross_validate(estimator=multi_output, cv=10, X=ind_cols_loc, y=dep_cols_loc)
    print("It's somewhat working!")

def main():
    # build_sequencer(True, False)
    # build_sequencer(True, True)
    # build_sequencer(False, False)
    # build_sequencer(False, True)
    # build_dataframes(True, False)
    # build_dataframes(True, True)

    build_dataframes(False, False)
    # build_dataframes(False, True)

    # build_model(False, False)

main()
