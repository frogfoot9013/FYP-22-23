import json
import string
import os
from gensim.models import Word2Vec
from DataLoaders import DataLoaderNoTags, DataLoaderWithTags

core_dir = os.getcwd()
preprocessed_dir = core_dir + "/Datasets/news_set_financial_preprocessed/"
sampled_dir = core_dir + "/Datasets/news_set_financial_sampled/"

def build_word_to_vec_models():
    dl_no_tags_sampled = DataLoaderNoTags(sampled_dir)
    model_no_tags_sampled = Word2Vec()
    model_no_tags_sampled.build_vocab(corpus_iterable=dl_no_tags_sampled)
    model_no_tags_sampled.train(corpus_iterable=dl_no_tags_sampled, epochs=model_no_tags_sampled.epochs, total_examples=model_no_tags_sampled.corpus_count)
    model_no_tags_sampled.save(core_dir+"/Models/sampled_dataset_no_tags_word2vec.model")
    dl_w_tags_sampled = DataLoaderWithTags(sampled_dir)
    model_w_tags_sampled = Word2Vec()
    model_w_tags_sampled.build_vocab(corpus_iterable=dl_w_tags_sampled)
    model_w_tags_sampled.train(corpus_iterable=dl_w_tags_sampled, epochs=model_w_tags_sampled.epochs, total_examples=model_w_tags_sampled.corpus_count)
    model_w_tags_sampled.save(core_dir+"/Models/sampled_dataset_w_tags_word2vec.model")
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

def main():
    build_word_to_vec_models()

main()
