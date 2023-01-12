from json import load as json_load
from json import dump as json_dump
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import contractions
import string
import os

# 'global' variables
core_dir = os.getcwd()
stop_words = set(stopwords.words('english'))

# functions to write:
# read in and pre-process single article
# write pre-processed article to some format
# do this across the entire set
# TODO: two (but possibly one) sets of preprocessing, one for unsupervised learning, one for supervised learning
def read_in_financial_news_single_article(input_directory, input_file):
    tgt_directory = os.path.basename(os.path.normpath(input_directory))
    # tgt_file = open(input_directory + "/" + input_file) # the actual path for properly reading in files
    # path for reading in test file
    tgt_file = open(core_dir + "/Datasets/test_article.json", "r")
    tgt_json = json_load(tgt_file)
    tgt_file.close()

    # read in required data points from json
    article_uuid = tgt_json["uuid"]
    article_time = tgt_json["published"]
    article_source = tgt_json["thread"]["site_full"]
    article_author = tgt_json["author"]
    article_entities = tgt_json["entities"]
    article_url = tgt_json["url"]
    article_title_not_preprocessed = tgt_json["title"]
    article_text_not_preprocessed = tgt_json["text"]

    # do pre-processing
    clean_headline = preprocess_text(article_title_not_preprocessed)
    clean_article = preprocess_text(article_text_not_preprocessed)
    
    # format json for writing to files
    output_json_classifying = {'uuid': article_uuid, 'site': article_source, 'author': article_author, 'published': article_time, 'entities': article_entities, 'url': article_url, 'title': clean_headline, 'text': clean_article}
    # TODO: consider if clustering can work with same source data, but only pull necessary columns at run-time

    # write to files
    # sentiment analysis classifier
    fptr = open(core_dir + "/Datasets/test_article_preprocessed.json", "w")
    json_dump(output_json_classifying, fptr, ensure_ascii=False)
    fptr.close()

# iterating across directories and files of data set
def preprocess_financial_dataset(input_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            read_in_financial_news_single_article(subdir, file)

# function for preprocessing, incomplete
def preprocess_text(input_text):
    # clear whitespace
    input_text = input_text.strip()
    input_text = " ".join(input_text.split())
    # expand contractions
    input_text = contractions.fix(input_text)
    # segment into sentences
    sentences = sent_tokenize(input_text, language='english')
    sentence_words = [word_tokenize(sen, language='english') for sen in sentences]

    # part of speech tagging
    sentences_tagged = [pos_tag(sen) for sen in sentence_words]

    # stop-word removal
    sentences_without_stopwords = []
    for sen in sentences_tagged:
        sen_filtered = []
        for word in sen:
            if word[0] not in stop_words:
                sen_filtered.append(word)
        sentences_without_stopwords.append(sen_filtered)

    # punctuation removal
    sentences_without_punctuation = []
    for sen in sentences_without_stopwords:
        sen_filtered = []
        for word in sen:
            if word[0] not in string.punctuation:
                sen_filtered.append(word)
        sentences_without_punctuation.append(sen_filtered)
    
    # TODO: consider the need for lowercasing

    return sentences_without_punctuation

# for testing purposes
def main():
    # preprocess_financial_dataset(core_dir + "/Datasets/news_set_financial")
    read_in_financial_news_single_article(core_dir, "test_article.json")

main()