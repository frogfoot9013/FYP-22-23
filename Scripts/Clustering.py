import json
import os
from sklearn.cluster import KMeans
from DataLoaders import count_entities
import pickle
import numpy as np
import pandas as pd
import polars as pl
import scipy as sp
core_dir = os.getcwd()

entities_to_ignore = ["reuters", "trump"] # entities that occur far more than any other entities in the set, excluding them for the sake of efficiency
            
def count_ens_files(tgt_dir, cutoff_point):
    ens = {}
    output_files = []
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
                        if el["name"] not in entities_to_ignore and el["name"] not in ens.keys() and el["sentiment"] != "none": # disregard 'none' sentiment
                            new_item = {el["name"]: [f_name]}
                            ens.update(new_item)
                        elif el["name"] not in entities_to_ignore and el["name"] in ens.keys() and el["sentiment"] != "none":
                            ens[el["name"]].append(f_name)
    
    output_ens = []
    ens = sorted(ens.items(), key=lambda x: len(x[1]), reverse=True)
    i = 1
    for el in ens:
        output_ens.append(el[0])
        for fl in el[1]:
            if fl not in output_files:
                output_files.append(fl)
        i += 1
        if i > cutoff_point:
            break
    return output_ens, output_files

def write_ens_files_lists(entities, files, name, amt):
    pickle.dump(entities, open(core_dir+"/Models/list_"+ name + "_entities_"+str(amt)+".pkl", "wb"))
    pickle.dump(files, open(core_dir+"/Models/list_" + name + "_files_"+str(amt)+".pkl", "wb"))
    print("Lists written successfully!")

def read_ens_files_lists(name, amt):
    output_entities = pickle.load(open(core_dir+"/Models/list_" + name + "_entities_" + str(amt)+".pkl", "rb"))
    output_files = pickle.load(open(core_dir+"/Models/list_" + name + "_files_" + str(amt) + ".pkl", "rb"))
    if output_entities != None and output_files != None:
        print("Loading successful!")
    return output_entities, output_files

def build_array(entities, files):
    output_df = pd.DataFrame(index=entities, columns=files)
    for f in files:
        fptr = open(f, "r")
        file_json = json.load(fptr)
        fptr.close()
        file_entities = file_json["entities"]
        for key, val in file_entities.items(): # consider changing to iterate over everything instead of what's in the file
            for el in val:
                if el["name"] in entities and el["sentiment"] != 'none':
                    sen = el["sentiment"]
                    if sen == 'positive':
                        output_df.at[el["name"], f] = 5
                    elif sen == "negative":
                        output_df.at[el["name"], f] = 4
                    elif sen == "neutral":
                        output_df.at[el["name"], f] = 1
                    else:
                        output_df.at[el["name"], f] = -1
    output_df.fillna(0, inplace=True)
    return output_df

def make_cluster(input_df):
    keys = input_df.index
    k_val = 4
    km = KMeans(n_clusters=k_val, n_init=1, max_iter=200).fit(X=input_df)
    labels = km.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = [keys[i]]
        else:
            clusters[label].append([keys[i]])
    for el in clusters:
        print("Elements in Cluster ", el, ":")
        print(clusters[el])
    print("This seems to be working.")

def main():
    da_entities, da_files = read_ens_files_lists("full_dataset", 1000)
    da_array = build_array(da_entities, da_files)
    make_cluster(da_array)
    # da_entities, da_files = count_ens_files(core_dir+"/Datasets/news_set_financial_preprocessed", 1000)
    # write_ens_files_lists(da_entities, da_files, "full_dataset", 1000)

main()
