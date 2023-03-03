import string
import os
import json
from gensim.models import Word2Vec
import DataLoaders
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GridSearchCV
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
                        if el["name"] not in ens.keys() and el["sentiment"] != "none": # disregard 'none' sentiment
                            new_item = {el["name"]: 1}
                            ens.update(new_item)
                        elif el["name"] in ens.keys() and el["sentiment"] != "none":
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
        print("New seq_len: ", new_val)
        
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
def build_dataframes(tgt_dataset_dir, df_identifier, use_tags, sq_ln, most_ens):
    ind_cols_df_name = core_dir + "/Models/df_ind_cols_" + df_identifier + "_" + str(most_ens)
    dep_cols_df_name = core_dir + "/Models/df_dep_col_" + df_identifier + "_" + str(most_ens)
    sq_txt_name = ''
    if use_tags:
        ind_cols_df_name += "_w_tags.pkl"
        dep_cols_df_name += "_w_tags.pkl"
        sq_txt_name = core_dir + "/Models/sequencer_full_dataset_w_tags.pkl"
    else:
        ind_cols_df_name += "_no_tags.pkl"
        dep_cols_df_name += "_no_tags.pkl"
        sq_txt_name = core_dir + "/Models/sequencer_full_dataset_no_tags.pkl"
    
    sq_sites = pickle.load(open(core_dir + "/Models/sequencer_sites_full.pkl", "rb"))
    sq_entities = pickle.load(open(core_dir + "/Models/sequencer_entities_full.pkl", "rb"))
    sq_text = pickle.load(open(sq_txt_name, "rb"))
    sq_entities.change_seq_len(sq_ln)
    sq_text.change_seq_len(sq_ln)
    relevant_entities = count_entities(tgt_dataset_dir, most_ens)
    print("This is working!")

    key_col = []; uuid_col = []; site_col = []; 
    title_col = []; text_col = []; entities_col = []; sentiments_col = []
    i = 1
    for dirpath, subdirs, files in os.walk(tgt_dataset_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            file_entities = file_json["entities"]
            el_count = sum(len(el) for el in file_entities.values())
            if el_count < 1: # skip document if no entities are present
                continue
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
    

    pickle.dump(df_ind_cols, open(ind_cols_df_name, "wb"))
    pickle.dump(df_dep_col, open(dep_cols_df_name, "wb"))
    print("Columns constructed and written!")

# Function to load DataFrames that already exist
def load_dataframes(identifier, use_tags, most_ens):
    ind_cols_file_name = core_dir + "/Models/df_ind_cols_" + identifier + "_" + str(most_ens)
    dep_cols_file_name = core_dir + "/Models/df_dep_col_" + identifier + "_" + str(most_ens)
    if use_tags == True:
        ind_cols_file_name += "_w_tags.pkl"
        dep_cols_file_name += "_w_tags.pkl"
    else:
        ind_cols_file_name += "_no_tags.pkl"
        dep_cols_file_name += "_no_tags.pkl"
    
    if os.path.isfile(ind_cols_file_name) == False or os.path.isfile(dep_cols_file_name) == False:
        print("DataFrames don't exist!")
        return None
    else:
        ind_cols_df = pickle.load(open(ind_cols_file_name, "rb"))
        dep_cols_df = pickle.load(open(dep_cols_file_name, "rb"))
        return ind_cols_df, dep_cols_df

def build_model(ind_cols, dep_col):
    dep_col_names = 'Sentiment'
    # ind_cols_names = ['Site', 'Title', 'Text', 'Name']
    # ind_cols_names = ['Title', 'Text', 'Name']
    ind_cols_names = ['Title', 'Text', 'Name']
    ind_cols = ind_cols.loc[:,ind_cols_names]
    ind_cols = ind_cols.values.tolist()
    ind_cols = np.array(ind_cols)
    print(ind_cols.shape)
    ind_cols = ind_cols.reshape(ind_cols.shape[0],-1) # is bodge, not sure if scalable at all
    print(ind_cols.shape)
    dep_col = list(dep_col["Sentiment"])
    svc_model = SVC(cache_size=500, kernel='linear', decision_function_shape='ovo')
    svc_results = cross_validate(svc_model, ind_cols, dep_col, scoring='balanced_accuracy', cv=2)
    # svc_results = GridSearchCV(estimator=svc_model, cv=2, n_jobs=-1, param_grid=[{'kernel':['poly'], 'degree':range(3,5), 'gamma':['scale', 'auto'], 'decision_function_shape':['ovo','ovr']}, {'kernel':['linear'], 'decision_function_shape':['ovo','ovr']}, {'kernel':['rbf', 'sigmoid'], 'degree':range(3,5), 'gamma':['scale','auto'], 'decision_function_shape':['ovo','ovr']}], scoring='balanced_accuracy')
    # svc_results.fit(ind_cols, dep_col)
    print("SVC fitted!")
    print(svc_results)
    '''print("Best score: ", svc_results.best_score_)
    print("Best params: ", svc_results.best_params_)'''

# simple function to construct a w2v model
def build_w2v_model(dl):
    w2v = Word2Vec()
    w2v.build_vocab(corpus_iterable=dl)
    w2v.train(corpus_iterable=dl, epochs=w2v.epochs, total_examples=w2v.corpus_count)
    return w2v

# function used to build more sequencers
def build_large_sequencers():
    tgt_files_dir = core_dir + "/Datasets/news_set_financial_preprocessed"
    s_len = 120

    dl_entities = DataLoaders.DataLoaderEntities(tgt_files_dir)
    w2v_entities = build_w2v_model(dl_entities)
    sq_entities = Sequencer(all_words=list(dl_entities), seq_len=s_len, embedding_matrix=w2v_entities.wv)
    pickle.dump(sq_entities, open(core_dir + "/Models/sequencer_entities_full.pkl", "wb"))

    dl_sites = DataLoaders.DataLoaderGeneric(tgt_files_dir, "site")
    w2v_sites = build_w2v_model(dl_sites)
    sq_sites = Sequencer(all_words=list(dl_sites), seq_len=s_len, embedding_matrix=w2v_sites.wv)
    pickle.dump(sq_sites, open(core_dir + "/Models/sequencer_sites_full.pkl", "wb"))
    print("Sequencers constructed!")

# This has been used to test on a much smaller scale, pending a reconstruction of earlier functions
# Needs refactoring to be more portable like the functions
def construct_classifier_model(use_tags):
    tgt_files_dir = core_dir + "/Datasets/news_set_financial_truncated/"
    s_len = 120 # Note: all vectors must be of an identical length for SVC
    relevant_entities = count_entities(tgt_files_dir, 500)
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
    sq_text = Sequencer(all_words=dl_lst_fixed, seq_len=s_len, embedding_matrix=w2v_text.wv)
    '''sq_text = ''
    if use_tags == True:
        sq_text = pickle.load(open(core_dir+"/Models/sequencer_full_dataset_w_tags.pkl", "rb"))
    else:
        sq_text = pickle.load(open(core_dir+"/Models/sequencer_full_dataset_no_tags.pkl", "rb"))'''

    '''sq_entities = pickle.load(open(core_dir + "/Models/sequencer_entities_full.pkl", "rb"))
    sq_sites = pickle.load(open(core_dir + "/Models/sequencer_entities_full.pkl", "rb"))
    sq_text.change_seq_len(s_len)
    sq_entities.change_seq_len(s_len)
    sq_sites.change_seq_len(s_len)'''

    print("Sequencers loaded up!")

    dl_entities = DataLoaders.DataLoaderEntities(tgt_files_dir)
    w2v_entities = build_w2v_model(dl_entities)
    sq_entities = Sequencer(all_words=list(dl_entities), seq_len=s_len, embedding_matrix=w2v_entities.wv)

    dl_sites = DataLoaders.DataLoaderGeneric(tgt_files_dir, "site")
    w2v_sites = build_w2v_model(dl_sites)
    sq_sites = Sequencer(all_words=list(dl_sites), seq_len=s_len, embedding_matrix=w2v_sites.wv)

    key_col = []; site_col = []; 
    title_col = []; text_col = []; entities_col = []; sentiments_col = []
    i = 1
    for dirpath, subdirs, files in os.walk(tgt_files_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            file_entities = file_json["entities"]
            el_count = sum(len(el) for el in file_entities.values())
            if el_count < 1: # skip document if no entities are present
                continue
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
                            site_col.append(site_vector)
                            title_col.append(title_vector)
                            text_col.append(text_vector)
                            embedded_entity = sq_entities.text_to_vector(el["name"])
                            entities_col.append(embedded_entity)
                            sentiments_col.append(el["sentiment"])
    
    ind_col_data = {'Site': site_col, 'Title': title_col, 'Text': text_col, 'Name': entities_col}
    # ind_col_data = {'Title': title_col, 'Text': text_col, 'Name': entities_col}
    df_dep_col = pd.DataFrame(index=key_col, data={'Sentiment': sentiments_col})
    df_ind_cols = pd.DataFrame(index=key_col, data=ind_col_data)

    dep_col_names = 'Sentiment'
    ind_cols_names = ['Site', 'Title', 'Text', 'Name']
    # ind_cols_names = ['Title', 'Text', 'Name']
    df_ind_cols = df_ind_cols.loc[:,ind_cols_names]
    df_ind_cols = df_ind_cols.values.tolist()
    df_ind_cols = np.array(df_ind_cols)
    df_ind_cols = df_ind_cols.reshape(df_ind_cols.shape[0],-1) # is bodge, not sure if scalable at all
    df_dep_col = list(df_dep_col["Sentiment"])
    svc_model = SVC(cache_size=500, kernel='linear', decision_function_shape='ovo')
    svc_results = cross_validate(svc_model, df_ind_cols, df_dep_col, scoring='balanced_accuracy', cv=2)
    # svc_results = GridSearchCV(estimator=svc_model, cv=2, n_jobs=-1, param_grid={'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'decision_function_shape':['ovo','ovr']}, scoring='balanced_accuracy')
    # svc_results.fit(df_ind_cols, df_dep_col)
    print("SVC fitted!")
    print(svc_results)
    # print("Best score: ", svc_results.best_score_)
    # print("Best params: ", svc_results.best_params_)
    '''tree_model = DecisionTreeClassifier()
    tree_results = cross_validate(tree_model, df_ind_cols, df_dep_col, scoring='balanced_accuracy', cv=2)
    tree_results = GridSearchCV(estimator=tree_model, cv=2, n_jobs=2, scoring='balanced_accuracy', param_grid={'criterion':['gini','entropy','log_loss'], 'class_weight':['balanced', None]})
    tree_results.fit(df_ind_cols, df_dep_col)
    print("Decision tree fitted!")
    print("Best score: ", tree_results.best_score_)
    print("Best params: ", tree_results.best_params_)'''

def main():
    # build_dataframes(tgt_dataset_dir=core_dir+"/Datasets/news_set_financial_truncated/", df_identifier="truncated", use_tags=True, sq_ln=120, most_ens=500)
    # ind_cols_df, dep_cols_df = load_dataframes("truncated", True, 500)
    # build_model(ind_cols_df, dep_cols_df)
    construct_classifier_model(True)
    # construct_classifier_model(False)

main()
