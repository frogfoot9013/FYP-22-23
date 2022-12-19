from json import load as json_load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


stop_words = set(stopwords.words('english'))

# functions to write:
# read in and pre-process single article
# write pre-processed article to some format
# do this across the entire set
def read_in_financial_news_single_article(input_article, input_location):
    tgt = input_location + "/" + input_article
    tgt_file = open(tgt)
    tgt_json = json_load(tgt_file)
    article_title_not_preprocessed = tgt_json["title"]
    output_url = tgt_json["url"]
    article_text_not_preprocessed = tgt_json["text"]

    # do pre-processing
    article_title_tokenised = word_tokenize(article_title_not_preprocessed)
    article_text_tokenised = word_tokenize(article_text_not_preprocessed)

    print(article_title_tokenised)
    print(article_text_tokenised)

    clean_headline = []
    clean_article = []
    for word in article_title_tokenised:
        if (word not in stop_words and word not in string.punctuation):
            clean_headline.append(word)

    for word in article_text_tokenised:
        if (word not in stop_words and word not in string.punctuation):
            clean_article.append(word)
    
    print(clean_headline)
    print(clean_article)



def main():
    file_directory = "../Datasets"
    input_file_name = "test_article.json"
    read_in_financial_news_single_article(input_file_name, file_directory)


main()