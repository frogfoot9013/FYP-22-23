import string
import os
from pathlib import Path
import random

core_dir = os.getcwd()
sample_size = 5_000

def randomly_sample_from_directory(input_directory):
    tgt_files = []
    targ_path = Path(input_directory).glob("**/*.json")
    for k, path in enumerate(targ_path):
        if k < sample_size:
            tgt_files.append(str(path))
        else:
            i = random.randint(0,k)
            if i < sample_size:
                tgt_files[i] = str(path)
    
    # write files
    for f_path in tgt_files:
        file_name = os.path.basename(os.path.normpath(f_path))
        dest_path = core_dir + "/Datasets/news_set_financial_sampled/" + file_name
        if ( os.path.exists(dest_path)): # consider the fact that there are identically-named files in different directories
            dest_path_without_extension, extension = os.path.splitext((dest_path))
            i = 1
            while os.path.exists(dest_path_without_extension + "_ " + str(i) + extension):
                i+=1
            # print(dest_path_without_extension + "_" + str(i) + extension)
            os.system("cp " + f_path + " " + dest_path_without_extension + "_" + str(i) + extension)
        else:
            # print(dest_path)
            os.system("cp " + f_path + " " + dest_path)


def main():
    randomly_sample_from_directory(core_dir + "/Datasets/news_set_financial_preprocessed/" + "directory_5")


main()
