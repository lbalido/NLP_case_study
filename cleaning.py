import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
from os import listdir
from os.path import isfile
import os


# function to read in file; returns a list of words
def read_files(directory, words_file):
    return [word for line in open(directory+ '/' + words_file, 'r') for word in line.split()]

# Clean file function
def clean_file(data):
    # remove punctuation 
    translator = str.maketrans('', '', string.punctuation)
    data_strip = [i.translate(translator) for i in data]
    
    # lowercase
    data_strip = [w.lower() for w in data_strip]

    # remove stop words
    stopwords_ = set(stopwords.words('english'))
    data_strip = [w for w in data_strip if not w in stopwords_]
    
    # remove spaces
    data_strip = [w for w in data_strip if w]
    
    # remove urls
    data_strip = [w for w in data_strip if 'http' not in w]
    
    return data_strip


def clean_files_directory(directory):
    documents = dict()
    onlyfiles = [f for f in listdir(directory) if isfile(directory + '/' + f)]
    
    for i in onlyfiles: 
        # read in file
        data = read_files(directory, i)

        # clean file
        cleaned = clean_file(data)

        # return dict with file as name and bag of words as values
        documents[i] = cleaned

    result = dict()

    # converting dictionary to pd.dataframe
    for key,value in documents.items():
        keys = list()
        keys.append(value)   
        result[key]=keys
    
    document_df = pd.DataFrame.from_dict(result, orient = 'index', columns = ['articles'])

    return document_df
