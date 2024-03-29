import numpy as np
from numpy.linalg import svd
import pandas as pd
import cleaning
from collections import Counter

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import operator

from sklearn.decomposition import NMF

def join_strings(df):
    df['articles_joined'] = df['articles'].apply(lambda x: ' '.join(x))

def word_count(df):
    df['word_count'] = df['articles'].apply(lambda x: len(x))
    total_count = df['word_count'].sum(axis=0)
    return total_count

def get_common_words(df):
    text_body = []
    text_body_exclusive = []
    for article in df['articles']:
        text_body += article
        text_body_exclusive += set(article)
    most_common = Counter(text_body).most_common(20)
    most_common_exc = Counter(text_body_exclusive).most_common(20)
    return most_common, most_common_exc

def vectorize(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['articles_joined'])
    words = vectorizer.get_feature_names()
    return vectorizer, X, words

def get_concepts(df,n):
    _, X, words = vectorize(df)

    nmf = NMF(n_components=n)
    nmf.fit(X.toarray())
    W = nmf.transform(X.toarray())
    H = nmf.components_

    concepts = {}
    for i in range(n):
        index_h = np.argsort(H[i,:])[-10:]
        most_common_words = [words[idx] for idx in index_h]
        concepts[i] = most_common_words

    return concepts

def get_idfs(df):
    vectorizer, X, words = vectorize(df)

    idfs = vectorizer.idf_
    words_idfs = list(zip(words, idfs))

    smallest = sorted(words_idfs, key = operator.itemgetter(1))[:20]
    smallest = pd.Series([features[1] for features in smallest], index = [features[0] for features in smallest])
    biggest = sorted(words_idfs, key=operator.itemgetter(1), reverse=True)[:20]
    biggest = sorted(biggest, key = operator.itemgetter(0))
    biggest = pd.Series([features[1] for features in biggest], index = [features[0] for features in biggest])

    return smallest, biggest

def dense_matrix():
    _, X, words = vectorize(df)
    u,s,v = svd(X.toarray())
    return v
