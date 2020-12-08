import numpy as np
import pandas as pd
import nltk
import pickle
import gensim

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

EMBEDDING_DIM = 100
maxlen = 700

def convertXToFormatAcceptable(data):
    X = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    for par in data['fulltext'].values:
        tmp = []
        sentences = nltk.sent_tokenize(par)
        for sent in sentences:
            sent = sent.lower()
            tokens = tokenizer.tokenize(sent)
            filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
            tmp.extend(filtered_words)
        X.append(tmp)
    return X

def get_weight_matrix(model, vocab):
    vocal_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix

def tokenizerX(X):
    w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    word_index = tokenizer.word_index
    X = pad_sequences(X, maxlen=maxlen)
    vocab_size = len(tokenizer.word_index) + 1
    return X, word_index, vocab_size, w2v_model, tokenizer

def data_processing(real, fake):
    try:
        unknown_publishers = []
        for index, row in enumerate(real.text.values):
            try:
                record = row.split(" -", maxsplit = 1)
                assert (len(record[0]) < 260)
            except:
                unknown_publishers.append(index)
        real.iloc[unknown_publishers].text
        publisher = []
        tmp = []
        for index, row in enumerate(real.text.values):
            if index in unknown_publishers:
                tmp.append(row)
                publisher.append("Unknown")
                continue
            record = row.split(" -", maxsplit=1)
            publisher.append(record[0])
            tmp.append(record[1])
        real['publisher'] = publisher
        real['text'] = tmp
        del publisher, tmp, record, unknown_publishers
    except:
        pass
    real = real.drop([index for index, text in enumerate(real.text.values) if str(text).strip() == ''], axis=0)
    fake = fake.drop([index for index, text in enumerate(fake.text.values) if str(fake).strip() == ''], axis=0)
    real['fulltext'] = real['title'] + ' ' + real['text']
    fake['fulltext'] = fake['title'] + ' ' + fake['text']
    try:
        real = real.drop(["subject", 'text', "date", "title", "publisher"], axis=1)
    except:
        real = real.drop(["subject", 'text', "date", "title"], axis=1)
    fake = fake.drop(["subject", 'text', "date", "title"], axis=1)
    data = real.append(fake, ignore_index=True)
    return data

def databefore(real):
    unknown_publishers = []
    for index, row in enumerate(real.text.values):
        try:
            record = row.split(" -", maxsplit = 1)
            assert (len(record[0]) < 260)
        except:
            unknown_publishers.append(index)
    real.iloc[unknown_publishers].text
    publisher = []
    tmp = []
    try:
        for index, row in enumerate(real.text.values):
            if index in unknown_publishers:
                tmp.append(row)
                publisher.append("Unknown")
                continue
            record = row.split(" -", maxsplit=1)
            publisher.append(record[0])
            tmp.append(record[1])
        real['publisher'] = publisher
        real['text'] = tmp
        del record
    except:
        pass
    del publisher, tmp, unknown_publishers
    real = real.drop([index for index, text in enumerate(real.text.values) if str(text).strip() == ''], axis=0)
    real['fulltext'] = real['title'] + ' ' + real['text']
    try:
        real = real.drop(["subject", 'text', "date", "title", "publisher"], axis=1)
    except:
        real = real.drop(["subject", 'text', "date", "title"], axis=1)
    return real

def getDataBeforeTrain(dataFile):
    data = databefore(dataFile)
    X = convertXToFormatAcceptable(data)
    tokenizerData.fit_on_texts(X)
    X = tokenizerData.texts_to_sequences(X)
    X_test = pad_sequences(X, maxlen=maxlen)
    return X_test

try:
    with open("data.pickle", "rb") as f:
        X, y, word_index, vocab_size, embedding_vectors, tokenizerData = pickle.load(f)
except:
    fake = pd.read_csv("./input/Fake.csv")
    real = pd.read_csv("./input/True.csv")
    real["impression"] = 1
    fake["impression"] = 0
    data = data_processing(real, fake)
    y = data["impression"].values
    X = convertXToFormatAcceptable(data)
    del data
    X, word_index, vocab_size, w2v_model, tokenizerData = tokenizerX(X)
    embedding_vectors = get_weight_matrix(w2v_model, word_index)

    with open("data.pickle", "wb") as f:
        pickle.dump((X, y, word_index, vocab_size, embedding_vectors, tokenizerData), f)

#Defining Neural Network
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 4353)
try:
    json_file = open('model.json', 'r')
    load_model = json_file.read()
    json_file.close()
    model = model_from_json(load_model)
    # load weights into new model
    model.load_weights("model.LSTM")
except:
    model = Sequential()
    # Non-trainable embeddidng layer
    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen,
                        trainable=False))
    # LSTM
    model.add(LSTM(units=128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train, validation_split=0.3, epochs=6)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to LSTM
    model.save_weights("model.LSTM")

#Train test
def trainLSTM(X_test, y_test):
    # Class 0 (Fake) if predicted prob < 0.5, else class 1 (Real)
    y_pred = (model.predict(X_test) >= 0.5).astype("int")
    score = round(accuracy_score(y_test, y_pred), 2)
    print("accuracy:" +  str(score))
    score = round(precision_score(y_test, y_pred), 2)
    print("precision:" + str(score))
    score = round(recall_score(y_test, y_pred), 2)
    print("recall:" + str(score))
    score = round(f1_score(y_test, y_pred), 2)
    print("f1:" + str(score))
    score = round(accuracy_score(y_test, y_pred), 2)
    if score >= 0.5:
        return 1
    else:
        return 0

trainLSTM(X_test, y_test)

def GetPred(X_test):
    try:
        json_file = open('model_LSTM.json', 'r')
        load_model = json_file.read()
        json_file.close()
        model = model_from_json(load_model)
        # load weights into new model
        model.load_weights("model_LSTM.LSTM")
    except:
        model = Sequential()
        # Non-trainable embeddidng layer
        model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen,
                            trainable=False))
        # LSTM
        model.add(LSTM(units=128))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        model.fit(X, y, validation_split=0.3, epochs=6)

        model_json = model.to_json()
        with open("model_LSTM.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to LSTM
        model.save_weights("model_LSTM.LSTM")
    y_pred = (model.predict(X_test) >= 0.5).astype("int")
    real = 0
    fake = 0
    for y_p in y_pred:
        if y_p == 1:
            real += 1
        else:
            fake += 1
    if real >= fake:
        return 1
    else:
        return 0
