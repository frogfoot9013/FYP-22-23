import json
import os


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
