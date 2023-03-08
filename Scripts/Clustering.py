import json
import os
from sklearn.cluster import KMeans
from DataLoaders import count_entities
import pickle
import numpy as np
import pandas as pd
import scipy as sp
core_dir = os.getcwd()

# this mght be a better implementation
def get_ens_files(tgt_dir, en_amt):
    tgt_entities = count_entities(tgt_dir, en_amt)
    output_df = pd.DataFrame(index=tgt_entities)
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            json_ens = file_json['entities']
            file_ens = {}
            for key, val in json_ens.items():
                for el in val:
                    if el["name"] in tgt_entities and el["sentiment"] != 'none':
                        file_ens.update({el["name"]: el["sentiment"]})
            if len(file_ens) > 0: # This whole section might need a different approach
                file_df = pd.DataFrame(index=tgt_entities, columns=[file_path])
                file_df.name = file_path
                for el in file_ens.keys():
                    sen = file_ens.get(el)
                    if sen == 'positive':
                        file_df.at[el,file_path] = 5
                    elif sen == 'negative':
                        file_df.at[el,file_path] = 4
                    elif sen == 'neutral':
                        file_df.at[el,file_path] = 1
                    else:
                        file_df.at[el,file_path] = -1
                if output_df.empty:
                    output_df = file_df
                else:
                    output_df = output_df.join(file_df) # This seems to be rather computationally-expensive
    print(output_df.shape)
    return output_df
            
# NOTE: these two methods are a first pass
def get_uuids(tgt_dir):
    output = []
    duplicates = 0
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            uuid = file_json['uuid']
            if uuid in output:
                duplicates += 1
            else:
                output.append(file_json['uuid'])
    print("Number of duplicates: ", duplicates)
    return output


def build_array(tgt_dir, els_amt):
    all_entities = count_entities(tgt_dir, els_amt)
    uuids = get_uuids(tgt_dir, all_entities)
    cluster_array = pd.DataFrame(index=all_entities, columns=uuids)
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            json_ens = file_json["entities"]
            file_uuid = file_json["uuid"]
            file_ens = {}
            for key, val in json_ens.items():
                for el in val:
                    if el["name"] in all_entities and el["sentiment"] != 'none':
                        file_ens.update({el["name"]: el["sentiment"]})
            if file_ens:
                print("We are here!")
            else:
                print("Oh no")
            for el in all_entities: # slow but ensures array has no empty elements
                if el in file_ens.keys():
                    sen = file_ens.get(el)
                    if sen == "positive":
                        cluster_array.loc[el, file_uuid] = 5
                    elif sen == "negative":
                        cluster_array.loc[el, file_uuid] = 4
                    elif sen == "neutral":
                        cluster_array.loc[el, file_uuid] = 1
                    else:
                        cluster_array.loc[el, file_uuid] = 0
                else:
                    cluster_array.loc[el, file_uuid] = 0
    return cluster_array

# Array composition
# one dimension corresponding to document, other to entity
# value is 1 or 0 depending on whether or not an entity is in a given document

def make_cluster():
    the_frame = get_ens_files(core_dir+"/Datasets/news_set_financial_truncated", 120)
    # the_frame = build_array(core_dir+"/Datasets/news_set_financial_truncated", 100)
    keys = the_frame.index
    k_val = 4
    km = KMeans(n_clusters=k_val, n_init=1, max_iter=100, random_state=1).fit(X=the_frame)
    labels = km.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = [keys[i]]
        else:
            clusters[label].append([keys[i]])
    print(clusters)
    print("This seems to be working.")


def main():
    test_frame = get_ens_files(core_dir+"/Datasets/news_set_financial_sampled", 1000)
    print("This works!")

main()