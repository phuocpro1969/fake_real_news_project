import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import load_model

seed = 4353

def final(X_data_full):
    def remove_punct(X_data_func):
        string1 = X_data_func.lower()
        translation_table = dict.fromkeys(map(ord, string.punctuation), ' ')
        string2 = string1.stranslate(translation_table)
        return string2

    X_data_full_clear_punct = []
    for i in range(len(X_data_full_clear_punct)):
        test_data = remove_punct(X_data_full[i])
        X_data_full_clear_punct.append(test_data)

    #remove stopwords
    def remove_stopwords(X_data_func):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        string = pattern.sub(' ', X_data_func)
        return  string

    X_data_full_clear_stopwords = []
    for i in range(len(X_data_full)):
        test_data = remove_stopwords(X_data_full[i])
        X_data_full_clear_stopwords.append(test_data)

    # tokenize
    def tokenize_words(X_data):
        words = nltk.word_tokenize(X_data)
        return  words

    X_data_full_tokenized_words = []
    for i in range(len(X_data_full)):
        test_data = tokenize_words(X_data_full[i])
        X_data_full_tokenized_words.append(test_data)

    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(X_data):
        words = lemmatizer.lemmatize(X_data)
        return  words

    X_data_full_lemmatized_words = []
    for i in range(len(X_data_full)):
        test_data = lemmatize_words(X_data_full[i])
        X_data_full_lemmatized_words.append(test_data)
    return  X_data_full_lemmatized_words

try:
    with open("data_2.pickle", "rb") as f:
        X, y, tfidf = pickle.load(f)
except:
    true = pd.read_csv('./input/True.csv')
    fake = pd.read_csv('./input/Fake.csv')

    true['impression']=1
    fake['impression']=0

    dataFile = pd.concat([true, fake], axis=0)
    dataFile['fulltext'] = dataFile.title + ' ' + dataFile.text
    data = dataFile[['fulltext', 'impression']]
    data = data.reset_index()
    data.drop(['index'], axis=1, inplace=True)
    X = data['fulltext']
    X = X.astype(str)
    X = final(X)
    y= data['impression']

    #tf -idf with bag of words model
    cv = CountVectorizer(max_features=1000)
    X_data_full_Vector = cv.fit_transform(X).toarray()

    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X_data_full_Vector).toarray()

    with open("data_2.pickle", "wb") as f:
        pickle.dump((X, y, tfidf), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= seed)
### NB
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
### RF
RFC=RandomForestClassifier(n_estimators= 10, random_state= seed)
RFC.fit(X_train, y_train)

def trainNB(X_test, y_test):
    print("Naive Bayes model")
    y_pred = MNB.predict(X_test)
    score = round(accuracy_score(y_test, y_pred), 2)
    print("accuracy:" + str(score))
    score = round(precision_score(y_test, y_pred), 2)
    print("precision:" + str(score))
    score = round(recall_score(y_test, y_pred), 2)
    print("recall:" + str(score))
    score = round(f1_score(y_test, y_pred), 2)
    print("f1:" + str(score))

def trainRB(X_test, y_test):
    print("RandomForest model")
    y_pred = RFC.predict(X_test)
    score = round(accuracy_score(y_test, y_pred), 2)
    print("accuracy:" + str(score))
    score = round(precision_score(y_test, y_pred), 2)
    print("precision:" + str(score))
    score = round(recall_score(y_test, y_pred), 2)
    print("recall:" + str(score))
    score = round(f1_score(y_test, y_pred), 2)
    print("f1:" + str(score))

TNB = trainNB(X_test, y_test)
TRB = trainRB(X_test, y_test)
