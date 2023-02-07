import json
import string
import os
from gensim.models import Word2Vec

core_dir = os.getcwd()
preprocessed_dir = core_dir + "/Datasets/news_set_financial_preprocessed/"
sampled_dir = core_dir + "/Datasets/news_set_financial_sampled/"

class DataLoaderNoTags(): # this class is an iterable used for reading in text data from the data set and removing the tags
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        for dirpath, subdirs, files in os.walk(self.file_path):
            for f in files:
                f_name = os.path.join(dirpath, f)
                fptr = open(f_name)
                file_json = json.load(fptr)
                fptr.close()
                title_json = file_json["title"]
                text_json = file_json["text"]
                combined_json = []
                for sen in title_json:
                    combined_json.append(sen)
                for sen in text_json:
                    combined_json.append(sen)
                for sen in combined_json:
                    fixed_sen = []
                    for word in sen:
                        fixed_sen.append(word[0])
                    yield fixed_sen


class DataLoaderWithTags(): # this class functions similarly to the above one, except it concatenates the tag onto the stemmed word
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        for dirpath, subdirs, files in os.walk(self.file_path):
            for f in files:
                f_name = os.path.join(dirpath, f)
                fptr = open(f_name)
                file_json = json.load(fptr)
                fptr.close()
                title_json = file_json["title"]
                text_json = file_json["text"]
                combined_json = []
                for sen in title_json:
                    combined_json.append(sen)
                for sen in text_json:
                    combined_json.append(sen)
                for sen in combined_json:
                    fixed_sen = []
                    for word in sen:
                        new_word = word[0]+word[1]
                        fixed_sen.append(new_word)
                    yield fixed_sen


def build_word_to_vec_models():
    dl_no_tags_sampled = DataLoaderNoTags(sampled_dir)
    model_no_tags_sampled = Word2Vec()
    model_no_tags_sampled.build_vocab(corpus_iterable=dl_no_tags_sampled)
    model_no_tags_sampled.train(corpus_iterable=dl_no_tags_sampled, epochs=model_no_tags_sampled.epochs, total_examples=model_no_tags_sampled.corpus_count)
    model_no_tags_sampled.save(core_dir+"/Models/sampled_dataset_no_tags.model")
    dl_w_tags_sampled = DataLoaderWithTags(sampled_dir)
    model_w_tags_sampled = Word2Vec()
    model_w_tags_sampled.build_vocab(corpus_iterable=dl_w_tags_sampled)
    model_w_tags_sampled.train(corpus_iterable=dl_w_tags_sampled, epochs=model_w_tags_sampled.epochs, total_examples=model_w_tags_sampled.corpus_count)
    model_w_tags_sampled.save(core_dir+"/Models/sampled_dataset_w_tags.model")
    full_dataset_no_tags_dl = DataLoaderNoTags(preprocessed_dir)
    model_no_tags = Word2Vec()
    model_no_tags.build_vocab(corpus_iterable=full_dataset_no_tags_dl)
    model_no_tags.train(corpus_iterable=full_dataset_no_tags_dl, epochs=model_no_tags.epochs, total_examples=model_no_tags.corpus_count)
    model_no_tags.save(core_dir+"/Models/full_dataset_no_tags_word2vec.model")
    full_dataset_w_tags_dl = DataLoaderWithTags(preprocessed_dir)
    model_w_tags = Word2Vec()
    model_w_tags.build_vocab(corpus_iterable=full_dataset_w_tags_dl)
    model_w_tags.train(corpus_iterable=full_dataset_w_tags_dl, epochs=model_w_tags.epochs, total_examples=model_w_tags.corpus_count)
    model_w_tags.save(core_dir+"/Models/full_dataset_w_tags_word2vec.model")

'''def main():
    build_word_to_vec_models()

main()'''
