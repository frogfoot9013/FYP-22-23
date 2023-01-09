from json import load as json_load
from json import dump as json_dump
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

core_dir = os.getcwd()
stop_words = set(stopwords.words('english'))

# functions to write:
# read in and pre-process single article
# write pre-processed article to some format
# do this across the entire set
def read_in_financial_news_single_article(input_directory, input_file):
    tgt_directory = os.path.basename(os.path.normpath(input_directory))
    tgt_file = open(input_directory + "/" + input_file)
    # for purposes of testing default writing
    # tgt_file = open(core_dir + "/Datasets/test_article.json", "r")
    tgt_json = json_load(tgt_file)
    tgt_file.close()
    article_title_not_preprocessed = tgt_json["title"]
    output_url = tgt_json["url"]
    article_text_not_preprocessed = tgt_json["text"]

    # do pre-processing
    article_title_tokenised = word_tokenize(article_title_not_preprocessed)
    article_text_tokenised = word_tokenize(article_text_not_preprocessed)

    clean_headline = []
    clean_article = []
    for word in article_title_tokenised:
        if (word.lower() not in stop_words and word.lower() not in string.punctuation):
            clean_headline.append(word.lower())

    for word in article_text_tokenised:
        if (word.lower() not in stop_words and word.lower() not in string.punctuation):
            clean_article.append(word.lower())
    
    # append to text
    tgt_json["url"] = clean_headline
    tgt_json["text"] = clean_article
    # write to file
    fptr = open(core_dir + "/Datasets/news_set_financial_preprocessed/" + tgt_directory + "/" + input_file, "w")
    json_dump(tgt_json, fptr)
    fptr.close()
    # first pass

# iterating across directories and files of data set
def preprocess_financial_dataset(input_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            # print(os.path.join(subdir, file)) # replace with call for reading in article
            read_in_financial_news_single_article(subdir, file)


# for testing purposes
def main():
    preprocess_financial_dataset(core_dir + "/Datasets/news_set_financial")
    #read_in_financial_news_single_article(core_dir, "test_article.json")

main()