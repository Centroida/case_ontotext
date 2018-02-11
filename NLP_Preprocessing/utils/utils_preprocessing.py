# -*- coding: iso-8859-15 -*-
import unicodedata
import nltk
from nltk.stem.porter import PorterStemmer
import itertools
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tqdm import tqdm
import glob
import multiprocessing
from functools import partial
from tqdm import *
import ntpath
import os
import csv 
def preprocess_line(line, stopwords):
    org_1 = line["company1"].lower()
    org_2 = line["company2"].lower()
    snippet = line["snippet"].lower()

    snippet = snippet.replace(org_1, " <firstorganization> ")
    snippet = snippet.replace(org_2, " <secondorganization> ")
    snippet = re.sub(r"\d+", r"<number>", snippet)
    snippet = re.sub(r"\.", r"<eol>", snippet)
    snippet = re.sub(r"end", r"", snippet)
    snippet = re.sub(r"(?<=[\w>])<eol>", r" <eol>", snippet)
    snippet = re.sub(r"(?![\w<])<eol>", r"<eol> ", snippet)
    snippet = re.sub(r"(?<=')s", r" <owns>", snippet)
    snippet = re.sub(r"(?<=')es", r" <owns>", snippet)
    snippet = re.sub(r"\w{0,}<firstorganization>\w{0,}", "<firstorganization>", snippet)
    snippet = re.sub(r"\w{0,}<secondorganization>\w{0,}", "<secondorganization>", snippet)
    snippet = re.sub(r"-owned", " <owns>", snippet)
    snippet = re.sub(r"-owning", " <owns>", snippet)
    tokenizer = RegexpTokenizer(r"[A-za-z<>&\-]+")
    snippet = tokenizer.tokenize(snippet)
    snippet = [word for word in snippet if word not in stopwords]
    return " ".join(snippet)

def path_leaf(path):
    """get filename from path"""
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_stopwords(path):
    stopwords = []
    with open(path, "r") as f:
        for line in f.readlines():
            stopwords.append(line.rstrip())

    return stopwords

def load_file_pandas(path, columns, out_path):
    """loads a csv file in a pandas dataframe and selects only the 
    columns we are interested in.
    Can pass both array of indices of columns or array of column
    names as strings to the columns parameter"""
    stopwords = get_stopwords("data/raw_data/stopwords.txt")

    df = pd.read_csv(open(path, "rU", encoding="utf-8"), 
                       encoding="utf-8", 
                       engine="c",
                       index_col=False,
                       header=0,
                       error_bad_lines=False)
    print("The shape of the dataframe is: ", df.shape)
    df["snippet"] = df.apply(lambda row: preprocess_line(row, stopwords), axis=1)

    with open(out_path, "a") as f:
        for idx, line in df.iterrows():
            snippet = line["snippet"]
            f.write(snippet + '\n')


def preprare_train_test_data(path, columns, out_path_train, out_path_test):
    stopwords = get_stopwords("data/raw_data/stopwords.txt")

    df = pd.read_csv(open(path, "rU", encoding="utf-8"), 
                       encoding="utf-8", 
                       engine="c",
                       index_col=False,
                       header=0,
                       error_bad_lines=False)
    print("The shape of the dataframe is: ", df.shape)
    df["snippet"] = df.apply(lambda row: preprocess_line(row, stopwords), axis=1)

    f_train = open(out_path_train, "a")
    f_test = open(out_path_test, "a")

    for idx, line in df.iterrows():
        label = "0"
            
        if line["is_parent"] == True:
            label = "1"

        snippet = line["snippet"]
        
        if idx < 71560:
            f_train.write(snippet + " " + label + '\n')
        else:
            f_test.write(snippet + " " + label + '\n')

def iterate_dictionary(path):
    """iterates through all kinds of files in a directory by using the 
    glob package and leaves directories alone"""

    if path[-1] != "/":
        path += "/"

    files_only = list(filter(lambda obj: os.path.isfile(obj), glob.glob(path + "*")))
    
    return files_only

def fix_file_lines(path, out_path):
    with open(path, "r") as f:
        text = f.read()
        lines = re.compile("#+END#+\"{0,}").split(text)
        with open(out_path, "a") as f_1:
            for line in lines:
                line = line.replace("\n", "")
                f_1.write(line + "\"" + "\n")