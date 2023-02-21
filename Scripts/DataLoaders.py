import json
import os

# TODO: Consider a way to refactor to reduce duplicate code

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

class DataLoaderEntities():
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        for dirpath, subdirs, files in os.walk(self.file_path):
            for f in files:
                f_name = os.path.join(dirpath, f)
                fptr = open(f_name)
                file_json = json.load(fptr)
                fptr.close()
                entities_json = file_json["entities"]
                all_entities = []
                for key, value in entities_json.items():
                    for el in value:
                        all_entities.append(el["name"])
                for el in all_entities:
                    yield el

# May serve as a basis for refactoring the other DataLoader classes
class DataLoaderGeneric():
    def __init__(self, file_path, tgt_tag):
        self.file_path = file_path
        self.tgt_tag = tgt_tag
    
    def __iter__(self):
        for dirpath, subdirs, files in os.walk(self.file_path):
            for f in files:
                f_name = os.path.join(dirpath, f)
                fptr = open(f_name)
                file_json = json.load(fptr)
                fptr.close()
                tgt_json = file_json[self.tgt_tag]
                yield tgt_json
