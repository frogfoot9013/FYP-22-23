from json import load as json_load
from json import dump as json_dump
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os

core_dir = os.getcwd()
stop_words = set(stopwords.words('english'))
word_set = set(words.words())
lemmatizer = WordNetLemmatizer()

# functions to write:
# read in and pre-process single article
# write pre-processed article to some format
# do this across the entire set
def read_in_financial_news_single_article(input_directory, input_file):
    tgt_directory = os.path.basename(os.path.normpath(input_directory))
    # tgt_file = open(input_directory + "/" + input_file)
    # for purposes of testing default writing
    tgt_file = open(core_dir + "/Datasets/test_article.json", "r")
    tgt_json = json_load(tgt_file)
    tgt_file.close()
    article_title_not_preprocessed = tgt_json["title"]
    output_url = tgt_json["url"]
    article_text_not_preprocessed = tgt_json["text"]

    # do pre-processing # maybe rework into function

    clean_headline = preprocess_text(article_title_not_preprocessed)
    clean_article = preprocess_text(article_text_not_preprocessed)
    
    # append to text
    tgt_json["title"] = clean_headline
    tgt_json["text"] = clean_article
    # write to file
    # for purposes of writing to test file
    fptr = open(core_dir + "/Datasets/test_article_preprocessed.json", "w")
    # fptr = open(core_dir + "/Datasets/news_set_financial_preprocessed/" + tgt_directory + "/" + input_file, "w")
    json_dump(tgt_json, fptr)
    fptr.close()


# iterating across directories and files of data set
def preprocess_financial_dataset(input_directory):
    for subdir, dirs, files in os.walk(input_directory):
        for file in files:
            read_in_financial_news_single_article(subdir, file)

# function for preprocessing, incomplete
def preprocess_text(input_text):
    input_text_tokenised = word_tokenize(input_text)
    input_text_tokens_lowercased = [token.lower() for token in input_text_tokenised]
    input_text_tokens_without_punctuation = [token for token in input_text_tokens_lowercased if token not in string.punctuation]
    input_text_tokens_without_stopwords = [token for token in tokens if token not in stop_words]
    return input_text_tokens_without_stopwords

# for testing purposes
def main():
    # preprocess_financial_dataset(core_dir + "/Datasets/news_set_financial")
    read_in_financial_news_single_article(core_dir, "test_article.json")

main()