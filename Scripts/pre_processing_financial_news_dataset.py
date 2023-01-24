from json import load as json_load
from json import dump as json_dump
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import contractions
import string
import os

# 'global' variables
core_dir = os.getcwd()
stop_words = set(stopwords.words('english'))
total_file_count = 306243 # might be useful for sampling the financial news dataset
punct_list = string.punctuation + '”' + '”' + '“' + '’' + '‘' # string.punctuation alone was not entirely satisfactory
stemmer = PorterStemmer()

# functions to write:
# read in and pre-process single article
# write pre-processed article to some format
# do this across the entire set
def read_in_financial_news_single_article(input_directory, input_file):
    fptr = open(input_directory + "/" + input_file) # the actual path for properly reading in files
    # path for reading in test file
    # fptr = open(core_dir + "/Datasets/test_article.json", "r")
    tgt_json = json_load(fptr)
    fptr.close()

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
    output_json = {'uuid': article_uuid, 'site': article_source, 'author': article_author, 'published': article_time, 'entities': article_entities, 'url': article_url, 'title': clean_headline, 'text': clean_article}

    # write to files
    # get preprocessed directory
    directory_no = os.path.basename(os.path.normpath(input_directory))
    fptr = open(core_dir + "/Datasets/news_set_financial_preprocessed/" + directory_no + "/" + input_file, "w")
    # for testing
    # fptr = open(core_dir + "/Datasets/test_article_preprocessed.json", "w")
    json_dump(output_json, fptr, ensure_ascii=False)
    fptr.close()

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
    # TODO: consider how improved named-entity-recognition could be applied, i.e. treating '10 Downing Street' as a single entity
    sentences_tagged = [pos_tag(sen) for sen in sentence_words]

    # consider the need for lowercasing
    sentences_lowercased = []
    for sen in sentences_tagged:
        sen_lowercased = []
        for word in sen:
            temp = list(word)
            temp[0] = temp[0].lower()
            sen_lowercased.append(tuple(temp))
        sentences_lowercased.append(sen_lowercased)

    # stop-word removal
    sentences_without_stopwords = []
    for sen in sentences_lowercased:
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
            if word[0] not in punct_list:
                sen_filtered.append(word)
        sentences_without_punctuation.append(sen_filtered)

    # consider the need for removing numerals
    sentences_without_numerals = []
    for sen in sentences_without_punctuation:
        sen_filtered = []
        for word in sen:
            if word[1] != "CD": # CD is pos-tag for a cardinal digit
                sen_filtered.append(word)
        sentences_without_numerals.append(sen_filtered)

    # Performing of stemming, because lemmatisation has proven too time-consuming
    # Stemming being carried out after tagging and stop-word removal so information pertinent to part-of-speech tagging is not lost
    sentences_stemmed = []
    for sen in sentences_without_numerals:
        sen_filtered = []
        for word in sen:
            temp = list(word)
            temp[0] = stemmer.stem(temp[0])
            sen_filtered.append(tuple(temp))
        sentences_stemmed.append(sen_filtered)

    output = sentences_stemmed
    return output

# iterating across directories and files of data set
# TODO: consider the use of sampling
def preprocess_financial_dataset(input_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            read_in_financial_news_single_article(subdir, file)

# for testing purposes
def main():
    preprocess_financial_dataset(core_dir + "/Datasets/news_set_financial/directory_1")
    # read_in_financial_news_single_article(core_dir, "test_article.json")

main()
