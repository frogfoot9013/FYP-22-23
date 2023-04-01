from DataLoaders import count_entities
import json
import os
core_dir = os.getcwd()

def count_articles(tgt_dir):
    count = 0
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            count += 1
    return count

def find_longest_article(tgt_dir, lim):
    longest_title = 0
    longest_text = 0
    total_title_len = 0
    total_text_len = 0
    article_count = 0
    articles_above_lim = 0
    articles_below_lim = 0
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            article_count += 1
            title_len = 0
            text_len = 0
            file_title = file_json['title']
            file_text = file_json['text']
            for sen in file_title:
                for word in sen:
                    title_len += 1
            total_title_len += title_len
            for sen in file_text:
                for word in sen:
                    text_len += 1
            if text_len > lim:
                articles_above_lim += 1
            else:
                articles_below_lim += 1
            total_text_len += text_len
            if title_len > longest_title:
                longest_title = title_len
            if text_len > longest_text:
                longest_text = text_len
    avg_title_len = total_title_len / article_count
    avg_text_len = total_text_len / article_count
    return longest_title, longest_text, avg_title_len, avg_text_len, articles_above_lim, 

def count_sentiments(tgt_dir):
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    positive_lst = []
    neutral_lst = []
    negative_lst = []
    for dirpath, subdirs, files in os.walk(tgt_dir):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            file_entities = file_json["entities"]
            for key, value in file_entities.items():
                if (len(value) > 1):
                    for el in value:
                        if el["sentiment"] == 'positive':
                            positive_count += 1
                            if file_path not in positive_lst:
                                positive_lst.append(file_path)
                        elif el["sentiment"] == 'neutral':
                            neutral_count += 1
                            if file_path not in neutral_lst:
                                neutral_lst.append(file_path)
                        elif el["sentiment"] == 'negative':
                            negative_count += 1
                            if file_path not in negative_lst:
                                negative_lst.append(file_path)
    print("Positive files: ", len(positive_lst))
    print("Neutral files: ", len(neutral_lst))
    print("Negative files: ", len(negative_lst))
    return positive_count, neutral_count, negative_count

def da_mission(en_1, en_2):
    for dirpath, subdirs, files in os.walk(core_dir + "/Datasets/news_set_financial_preprocessed"):
        for f in files:
            file_path = os.path.join(dirpath, f)
            fptr = open(file_path)
            file_json = json.load(fptr)
            fptr.close()
            file_entities = file_json["entities"]
            file_title = file_json["title"]
            found_en_1 = False
            found_en_2 = False
            for key, value in file_entities.items():
                if (len(value) > 1):
                    for el in value:
                        if el["name"] == en_1:
                            found_en_1 = True
                        elif el["name"] == en_2:
                            found_en_2 = True
                        if found_en_1 == True and found_en_2 == True:
                            break
            if found_en_1 == True and found_en_2 == True:
                print(file_title)

'''da_count = count_articles(core_dir+"/Datasets/news_set_financial_sampled")
print(da_count)

title_max, text_max, title_avg, text_avg, amt, amt_2 = find_longest_article(core_dir + "/Datasets/news_set_financial_sampled", 350)
print("Longest title: ", title_max, "\nLongest text: ", text_max, "\nAverage title length: ", title_avg, "\nAverage Text length: ", text_avg)
print("Articles over length: ", amt)
print("Articles under length: ", amt_2)'''

'''poss, neut, negt = count_sentiments(core_dir + "/Datasets/news_set_financial_preprocessed")
print("Positive: ", poss, ", Neutral: ", neut, " Negative: ", negt)

# poss, neut, negt = count_sentiments(core_dir + "/Datasets/news_set_financial_sampled")
# print("Positive: ", poss, ", Neutral: ", neut, " Negative: ", negt)

def read_file_bodge(tgt_dir):
    count = 0
    fptr = open(tgt_dir, 'r')
    for line in fptr:
        if line == "0.0\n":
            count += 1
    fptr.close()
    return count'''

'''count_bisecting_k_means = read_file_bodge(core_dir + "/Scores/Clusters_bisecting_kmeans_500_1.txt")
count_k_means = read_file_bodge(core_dir + "/Scores/Clusters_kmeans_450.txt")
print("Bisecting K-Means:")
print("One-Element Clusters: ", count_bisecting_k_means)
print("Multi-Element Clusters: ", 500-count_bisecting_k_means)
print("\nK-Means:")
print("One-Element Clusters: ", count_k_means)
print("Multi-Element Clusters: ", 450-count_k_means)'''

da_mission("foreign ministry", "opcw")
