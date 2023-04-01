import json
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from DataLoaders import count_entities
import pickle
import numpy as np
import pandas as pd
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
        for key, val in file_entities.items():
            for el in val:
                if el["name"] in entities and el["sentiment"] != 'none':
                    sen = el["sentiment"]
                    if sen == 'positive':
                        output_df.at[el["name"], f] = 5
                    elif sen == 'negative':
                        output_df.at[el["name"], f] = 4
                    elif sen == 'neutral':
                        output_df.at[el["name"], f] = 1
                    else:
                        output_df.at[el["name"], f] = 0
    output_df.fillna(0, inplace=True) # Consider alternative means of making sparse
    return output_df

def make_elbow_graph(input_df, model_type):
    inertias = {}
    file_name = core_dir+"/Scores/elbow_graph_"
    if model_type == "kmeans":
        file_name += "kmeans.csv"
    else:
        file_name += "bisecting_kmeans.csv"
    for i in range(100, 650, 50):
        if model_type == "kmeans":
            model = KMeans(n_clusters=i, max_iter=30, n_init=10).fit(input_df)
        else: # bisecting
            model = BisectingKMeans(n_clusters=i, n_init=10, max_iter=30, bisecting_strategy='largest_cluster').fit(input_df)
        print("Iteration complete!\nInertia: ", model.inertia_)
        inertias[i] = model.inertia_
    print("Writing to file.")
    fptr = open(file_name, 'w')
    fptr.write("Clusters,Inertia\n")
    for key, val in inertias.items():
        fptr.write(str(key) + "," + str(val) + '\n')
    fptr.close()
    print("Successfully written to file!")

def make_km_cluster(input_df, k_val, is_bisecting):
    keys = input_df.index
    output_file = core_dir + "/Scores/Clusters"
    model = ''
    if is_bisecting == True:
        output_file += "_bisecting_kmeans_" + str(k_val) + ".txt"
        model = BisectingKMeans(n_clusters=k_val, max_iter=300, n_init=1, bisecting_strategy='largest_cluster').fit(input_df)
    else:
        output_file += "_kmeans_" + str(k_val) + ".txt"
        model = KMeans(n_clusters=k_val, n_init=10, max_iter=300).fit(X=input_df)
    labels = model.labels_
    clusters = {}
    inertias = np.zeros(k_val)
    tf = model.transform(input_df)
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = [keys[i]] # For creating list of elements in each cluster
        else:
            clusters[label].append(keys[i])
        inertias[label] += tf[i][label]**2 # for computing inertia of each cluster
    for i, el in enumerate(clusters): # display clusters
        print("Elements in Cluster ", i, ":")
        print(clusters[i])
        print("Inertia of Cluster ", i, ":")
        print(inertias[i])
    print("\nDa Whole Inertia: ", model.inertia_)
    print(sum(inertias))
    fptr = open(output_file, 'w')
    for i, el in enumerate(clusters):
        fptr.write("Elements of Cluster " + str(i) + ":\n")
        fptr.write(str(clusters[i]) + "\n")
        fptr.write("Inertia of Cluster " + str(i) + ":\n")
        fptr.write(str(inertias[i]) + "\n")
    fptr.close()
    print("This seems to be working.")

def make_agglo_cluster(input_df, threshold, link):
    keys = input_df.index
    model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=threshold, linkage=link).fit(input_df)
    print("Clustering complete!")
    labels = model.labels_
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = [keys[i]]
        else:
            clusters[label].append(keys[i])
    for i, el in enumerate(clusters):
        print("Elements in Cluster ", i, ":")
        print(clusters[i])
    one_els, many_els = count_cluster_distribution(clusters)
    if one_els + many_els == model.n_clusters_:
        print("This sanity check has passed.")
    else:
        print("This is a problem.")
    print("Number of clusters with one element: ", one_els)
    print("Number of clusters with more than one element: ", many_els)
    fptr = open(core_dir+"/Scores/Clusters/counts.csv", "a")
    fptr.write(str(one_els) + "," + str(many_els) + "\n")
    fptr.close()
    fptr = open(core_dir + "/Scores/Clusters_Agglomerative.txt", "w")
    for i, el in enumerate(clusters):
        fptr.write("Elements of Cluster " + str(i) + ":\n")
        fptr.write(str(clusters[i]) + "\n")
    fptr.close()


def agglo_cluster_scoring(input_df, threshold, link):
    model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=threshold, linkage=link).fit(input_df)
    labels = model.labels_
    clusters = model.n_clusters_
    sil_coef = 0
    if clusters > 1 and clusters < len(input_df.index):
        sil_coef = silhouette_score(input_df, labels)
    else:
        print("Silhouette coefficient inapplicable!")
    print("Clustering complete!")
    return sil_coef, clusters

def count_cluster_distribution(input_clusters):
    one_el_clusters = 0
    many_el_clusters = 0
    for i, el in enumerate(input_clusters):
        if len(input_clusters[i]) == 1:
            one_el_clusters += 1
        else: 
            many_el_clusters += 1
    return one_el_clusters, many_el_clusters

def silhouette_analysis():
    da_entities, da_files = read_ens_files_lists("full_dataset", 1000)
    da_array = build_array(da_entities, da_files)
    print("Beginning clustering!")
    linkages = ['ward', 'average', 'single', 'complete']
    thresholds = range(0, 330, 30)
    sils = [[0 for x in thresholds] for y in linkages]
    cluster_counts = [[0 for x in thresholds] for y in linkages]
    for i, el in enumerate(linkages):
        for j, val in enumerate(thresholds):
            sils[i][j], cluster_counts[i][j] = agglo_cluster_scoring(da_array, val, el)
    fptr = open(core_dir+"/Scores/agg_silhouette.csv", "w")
    fptr.write("Linkage,Distance Threshold,\nLinkage,")
    for i, el in enumerate(thresholds):
        fptr.write(str(el) + ",")
    fptr.write('\n')
    for i, el in enumerate(linkages):
        fptr.write(el + ",")
        for j, val in enumerate(thresholds):
            fptr.write(str(sils[i][j]) + ",")
        fptr.write("\n")
    fptr.close()
    fptr = open(core_dir + "/Scores/agg_cluster_counts.csv", "w")
    fptr.write("Linkage,Distance Threshold,\nLinkage,")
    for i, el in enumerate(thresholds):
        fptr.write(str(el) + ",")
    fptr.write('\n')
    for i, el in enumerate(linkages):
        fptr.write(el + ",")
        for j, val in enumerate(thresholds):
            fptr.write(str(cluster_counts[i][j]) + ",")
        fptr.write('\n')
    fptr.close()

def main():
    da_entities, da_files = read_ens_files_lists("full_dataset", 1000)
    da_array = build_array(da_entities, da_files)
    print("Beginning clustering!")
    # make_agglo_cluster(da_array, 60, 'ward')
    # make_km_cluster(da_array, 600, False)
    make_km_cluster(da_array, 500, True)
    # make_elbow_graph(da_array, "kmeans")
    # print("KMeans complete!")
    # make_elbow_graph(da_array, "bisecting_kmeans")
    # print("Bisecting KMeans complete!")
    # da_entities, da_files = count_ens_files(core_dir+"/Datasets/news_set_financial_preprocessed", 1000)
    # write_ens_files_lists(da_entities, da_files, "full_dataset", 1000)

main()
