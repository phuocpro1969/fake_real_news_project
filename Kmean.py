import numpy as np
import pandas as pd
import seaborn as sns # improve visuals
import re
import pickle
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short # Preprocesssing
from gensim.models import Word2Vec # Word2vec
from sklearn import cluster # Kmeans clustering
from sklearn import metrics # Metrics for evaluation
from sklearn.decomposition import PCA #PCA
from sklearn.manifold import TSNE #TSNE

def remove_URL(data):
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return  regex.sub(r'', data)

# Preprocessing functions to remove lowercase, links, whitespace, tags, numbers, punctuation, strip words
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, remove_URL, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]

def return_vector(x):
    try:
        return model[x]
    except:
        return np.zeros(100)

def sentence_vector(sentence):
    word_vectors = list(map(lambda x: return_vector(x), sentence))
    return np.average(word_vectors, axis=0).tolist()

try:
    with open("data_kmean.pickle", "rb") as f:
        X_np, model = pickle.load(f)
except:
    fake = pd.read_csv("./input/Fake.csv")
    real = pd.read_csv("./input/True.csv")
    real["impression"] = 1
    fake["impression"] = 0

    real['fulltext'] = real['title'] + ' ' + real['text']
    fake['fulltext'] = fake['title'] + ' ' + fake['text']

    dataFile = pd.concat([fake, real])
    dataFile = dataFile.sample(frac=1).reset_index(drop=True)
    dataFile = dataFile.drop(['title', 'text', 'subject', 'date'], axis = 1)

    processed_data = []

    for index, row in dataFile.iterrows():
        words_broken_up = preprocess_string(row['fulltext'], CUSTOM_FILTERS)
        if len(words_broken_up) > 0:
            processed_data.append(words_broken_up)

    model = Word2Vec(processed_data, min_count=1)

    X = []
    for data_x in processed_data:
        X.append(sentence_vector(data_x))
    X_np = np.array(X)

    with open("data_kmean.pickle", "wb") as f:
        pickle.dump((X_np, model), f)

#model Kmean
kmeans = cluster.KMeans(n_clusters=2, verbose=1)
kmeans.fit_predict(X_np)
def trainKmean(dataFile):
    fulltext = preprocess_string(dataFile, CUSTOM_FILTERS)
    fulltext = sentence_vector(fulltext)
    pred = kmeans.predict(np.array([fulltext]))
    del fulltext
    return pred

