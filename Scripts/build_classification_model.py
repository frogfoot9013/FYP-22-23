import string
import os
import json
from sklearn import svm
from gensim.models import Word2Vec
import DataLoaders
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
core_dir = os.getcwd()

# used to determine x most frequent entities in dataset, and create list of just these entities
def count_entities(tgt_dir, cutoff_point):
    ens = {}
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            f_name = os.path.join(dirpath, f)
            fptr = open(f_name)
            file_json = json.load(fptr)
            fptr.close()
            file_entities = file_json["entities"]
            if len(file_entities) < 1:
                print("This line of code is being reached")
                continue
            else:
                for key, value in file_entities.items():
                    for el in value:
                        if el["name"] not in ens.keys():
                            new_item = {el["name"]: 1}
                            ens.update(new_item)
                        else:
                            new_val = ens.get(el["name"]) + 1
                            ens.update({el["name"]: new_val})
    
    ens_sorted = sorted(ens.items(), key=lambda x:x[1], reverse=True)
    output = []
    i = 1
    for el in ens_sorted:
        temp = list(el)
        output.append(temp[0])
        i += 1
        if i > cutoff_point:
            break
    return output

# based off code from https://www.kaggle.com/code/mehmetlaudatekman/tutorial-word-embeddings-with-svm, will probably need additional reworking for completion
class Sequencer():
    def __init__(self, all_words, seq_len, embedding_matrix):
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        self.vocab = list(set(all_words))
    
    def change_seq_len(self, new_val):
        self.seq_len = new_val
        print("New seq_len: " + str(new_val))
        
    def text_to_vector(self, text):
        output_vec = []
        len_v = len(text)-1 if len(text) < self.seq_len else self.seq_len-1
        for el in text[:len_v]:
            try:
                output_vec.append(self.embed_matrix[el])
            except Exception as E:
                output_vec.append(np.zeros(100,))
        
        last_pieces = self.seq_len - len(output_vec)
        for i in range(last_pieces):
            output_vec.append(np.zeros(100,))
        return np.asarray(output_vec).flatten()

def fix_text_no_tags(input_text):
    output = []
    for sen in input_text:
        for word in sen:
            output.append(word[0])
    return output

def fix_text_w_tags(input_text):
    output = []
    for sen in input_text:
        for word in sen:
            new_word = word[0]+word[1]
            output.append(new_word)
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
        dl = list(DataLoaders.DataLoaderWithTags(dataset_directory))
        print("Tags will be used!")
    else:
        dl = list(DataLoaders.DataLoaderNoTags(dataset_directory))
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
    sq.change_seq_len(120)
    

    uuid_col = []
    author_col = []
    # site_col = []
    title_col = []
    text_col = []
    name_col = []
    sentiment_col = []
    test_sen_col = []
    i = 0 # use of i is to reduce execution time for my own sanity
    for dirpath, subdirs, files in os.walk(dataset_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            uuid_col.append(file_json["uuid"])
            # author_col.append(file_json["author"])
            # site_col.append(file_json["site"]) # Note: consider one-hot encoding this variable
            fixed_title = []
            fixed_text = []
            if use_tags == True:
                fixed_title = fix_text_w_tags(file_json["title"])
                fixed_text = fix_text_w_tags(file_json["text"])
            else:
                fixed_title = fix_text_no_tags(file_json["title"])
                fixed_text = fix_text_no_tags(file_json["text"])
            
            title_vector = sq.text_to_vector(fixed_title)
            title_col.append(title_vector.tolist())
            text_vector = sq.text_to_vector(fixed_text)
            text_col.append(text_vector.tolist())

            file_entities = file_json["entities"]
            if len(file_entities["persons"]) >= 1:
                name_col.append(file_entities["persons"][0]["name"])
                sentiment_col.append(file_entities["persons"][0]["sentiment"])
            elif len(file_entities["locations"]) >= 1:
                name_col.append(file_entities["locations"][0]["name"])
                sentiment_col.append(file_entities["locations"][0]["sentiment"])
            elif len(file_entities["organizations"]) >= 1:
                name_col.append(file_entities["organizations"][0]["name"])
                sentiment_col.append(file_entities["organizations"][0]["sentiment"])
            else:
                name_col.append("placeholder")
                sentiment_col.append("none")
            i += 1
            if i > 200:
                break
        if i > 200:
            print("This line of code is being reached!")
            break
        # NOTE: commented out until more progress made
            '''doc_names = []
            doc_sentiments = []
            for en in file_entities["persons"]:
                doc_names.append(en["name"])
                doc_sentiments.append(en["sentiment"])
            for en in file_entities["locations"]:
                doc_names.append(en["name"])
                doc_sentiments.append(en["sentiment"])
            for en in file_entities["organizations"]:
                doc_names.append(en["name"])
                doc_sentiments.append(en["sentiment"])
            name_col.append(doc_names)
            sentiment_col.append(doc_sentiments)'''
    # TODO: Work out a way to properly process the sentiments, separated by article
    

    # TODO: find a way to get sentiment into data frame
    da_data = {'Title': title_col, 'Text': text_col}
    print("Lengths:\nUUID: " + str(len(uuid_col)) + "\nSentiment: " + str(len(sentiment_col)) + "\nTitle: " + str(len(title_col)) + "\nText: " + str(len(text_col)) + "\nName: " + str(len(name_col)))
    df_dep_cols = pd.DataFrame(index=uuid_col, data={'Sentiment': sentiment_col})
    df_ind_cols = pd.DataFrame(index=uuid_col, data=da_data)
    

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
    ind_cols_names = ['Title', 'Text']
    ind_cols_loc = df_ind_cols.loc[:,ind_cols_names]
    dep_cols_loc = df_dep_cols.loc[:,dep_cols_names]
    print("This is reached!")
    temp = df_ind_cols.values.tolist()
    temp = np.array(temp)
    temp = temp.reshape(temp.shape[0],-1) # is bodge, not sure if scalable at all
    crude_model = SVC(cache_size=500)
    crude_model.fit(temp, list(df_dep_cols["Sentiment"]))
    print("It's somewhat working!")

# simple function to construct a w2v model
def build_w2v_model(dl):
    w2v = Word2Vec()
    w2v.build_vocab(corpus_iterable=dl)
    w2v.train(corpus_iterable=dl, epochs=w2v.epochs, total_examples=w2v.corpus_count)
    return w2v

# Function does all functionality of constructing elements of model in one go
# Needs refactoring to be more portable like the functions
def construct_classifier_model(use_tags):
    tgt_files_dir = core_dir + "/Datasets/news_set_financial_truncated/"
    relevant_entities = count_entities(tgt_files_dir, 120)
    dl_text = ''
    if use_tags == True:
        dl_text = DataLoaders.DataLoaderWithTags(tgt_files_dir)
    else:
        dl_text = DataLoaders.DataLoaderNoTags(tgt_files_dir)
    w2v_text = build_w2v_model(dl_text)

    dl_lst = list(dl_text)
    dl_lst_fixed = []
    for sen in dl_lst:
        for word in sen:
            dl_lst_fixed.append(word)
    sq_text = Sequencer(all_words=dl_lst_fixed, seq_len=50, embedding_matrix=w2v_text.wv)

    dl_entities = DataLoaders.DataLoaderEntities(tgt_files_dir)
    w2v_entities = build_w2v_model(dl_entities)
    sq_entities = Sequencer(all_words=list(dl_entities), seq_len=50, embedding_matrix=w2v_entities.wv)

    dl_sites = DataLoaders.DataLoaderGeneric(tgt_files_dir, "site")
    w2v_sites = build_w2v_model(dl_sites)
    sq_sites = Sequencer(all_words=list(dl_sites), seq_len=50, embedding_matrix=w2v_sites.wv)

    key_col = []; uuid_col = []; site_col = []; title_col = []; text_col = []; entities_col = []; sentiments_col = []
    i = 1
    for dirpath, subdirs, files in os.walk(tgt_files_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            # check if there are entities in document, if not, ignore
            file_entities = file_json["entities"]
            if len(file_entities.items()) < 1:
                print("We are now here!")
            fixed_title = ''
            tixed_text = ''
            if use_tags == True:
                fixed_title = fix_text_w_tags(file_json["title"])
                fixed_text = fix_text_w_tags(file_json["text"])
            else:
                fixed_title = fix_text_no_tags(file_json["title"])
                fixed_text = fix_text_no_tags(file_json["text"])
            site_vector = sq_sites.text_to_vector(file_json["site"])
            title_vector = sq_text.text_to_vector(fixed_title)
            text_vector = sq_text.text_to_vector(fixed_text)
            for key, value in file_entities.items():
                if (len(value) > 1):
                    for el in value:
                        if el["sentiment"] != "none" and el["name"] in relevant_entities:
                            key_col.append(i)
                            i += 1
                            uuid_col.append(file_json["uuid"])
                            site_col.append(site_vector)
                            title_col.append(title_vector)
                            text_col.append(text_vector)
                            embedded_entity = sq_entities.text_to_vector(el["name"])
                            entities_col.append(embedded_entity)
                            sentiments_col.append(el["sentiment"])
    
    ind_col_data = {'Site': site_col, 'Title': title_col, 'Text': text_col, 'Name': entities_col}
    df_dep_col = pd.DataFrame(index=key_col, data={'Sentiment': sentiments_col})
    df_ind_cols = pd.DataFrame(index=key_col, data=ind_col_data)

    dep_col_names = 'Sentiment'
    ind_cols_names = ['Site', 'Title', 'Text', 'Name']
    ind_cols_loc = df_ind_cols.loc[:,ind_cols_names]
    dep_col_loc = df_dep_col.loc[:,dep_col_names]
    temp = df_ind_cols.values.tolist()
    temp = np.array(temp)
    temp = temp.reshape(temp.shape[0],-1) # is bodge, not sure if scalable at all
    crude_model = SVC(cache_size=5000)
    crude_model.fit(temp, list(df_dep_col["Sentiment"]))
    cv_results = cross_validate(crude_model, temp, list(df_dep_col["Sentiment"]), cv=2)
    print("It's somewhat working!")
    print(cv_results)

def main():
    construct_classifier_model(True)
    construct_classifier_model(False)

main()
