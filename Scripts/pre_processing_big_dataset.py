from json import load as json_load
from json import dump as json_dump
from nltk.corpus import stopwords, words, wordnet
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk import edit_distance
from nltk import pos_tag, ne_chunk
import contractions
import string
import os

# 'global' variables
core_dir = os.getcwd()
stop_words = set(stopwords.words('english'))
word_set = set(words.words())
# lemmatiser = WordNetLemmatizer()

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
    
    # append to text, TODO: determine format of output for writing to file
    tgt_json["title"] = clean_headline
    tgt_json["text"] = clean_article
    # write to file
    # for purposes of writing to test file
    fptr = open(core_dir + "/Datasets/test_article_preprocessed.json", "w")
    # fptr = open(core_dir + "/Datasets/news_set_financial_preprocessed/" + tgt_directory + "/" + input_file, "w")
    json_dump(tgt_json, fptr, ensure_ascii=False)
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

    # account for typos (takes too long so commented out)
    # corrected_words = []
    # for sentence in sentence_words:
        # corrected_sentence = []
        # for word in sentence:
            # corrected_word = min(word_set, key=lambda x: edit_distance(x, word))
            # corrected_sentence.append(corrected_word)
        # corrected_words.append(corrected_sentence)

    # part of speech tagging
    sentences_tagged = [pos_tag(sen) for sen in sentence_words]
    # wordnet_tagged = [list(map(lambda x: (x[0], wn_pos_tag(x[1])), sen)) for sen in sentences_tagged] # only for lemmatising
    # TODO: Named-entity recognition
    # lemmatising, commented out because the time taken is very bad across the whole data set
    # sentences_lemmatised = []
    # for sen in wordnet_tagged:
    #    lemmatised_sen = []
    #    for word, tag in sen:
    #        if tag is None:
    #            lemmatised_sen.append(word)
    #        else:
    #            lemmatised_sen.append(lemmatiser.lemmatize(word, tag))
    #    sentences_lemmatised.append(lemmatised_sen)

    # TODO: ask questions about stop-word removal and punctuation removal vis-à-vis POS-tagging

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

    return sentences_without_punctuation

# function to convert pos_tag's tags to tags the WordNet Lemmatiser can use
def wn_pos_tag(input):
    if input.startswith('J'):
        return wordnet.ADJ
    elif input.startswith('V'):
        return wordnet.VERB
    elif input.startswith('N'):
        return wordnet.NOUN
    elif input.startswith('R'):
        return wordnet.ADV
    else:
        return None


# for testing purposes
def main():
    # preprocess_financial_dataset(core_dir + "/Datasets/news_set_financial")
    read_in_financial_news_single_article(core_dir, "test_article.json")
    # test solely of preprocessing
    sample_text = "Hello there, Anna isn't very happy because her project's code is a mess and taking forever to write. Anna's code will eventually be finished, and she'll be satisfied then."
    preprocessed_text = preprocess_text(sample_text)
    print(preprocessed_text)

main()