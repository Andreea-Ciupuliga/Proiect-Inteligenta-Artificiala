import nltk
import pandas as pd
import random
import numpy as np
from collections import Counter

import sklearn
from nltk.corpus import stopwords

import time
timp_inceput=time.time()

def tokenize(text):
    
    tokens = nltk.WordPunctTokenizer().tokenize(text)

    tokens = [w.lower() for w in tokens]

    cuvinte = [cuvant for cuvant in tokens if cuvant.isalpha()]

    stop_words = set(stopwords.words('italian'))
    cuvinte = [w for w in cuvinte if not w in stop_words]

    return cuvinte



def get_corpus_vocabulary(corpus):

    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def get_representation(toate_cuvintele, how_many):

    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):


    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features


def corpus_to_bow(corpus, wd2idx):


    all_features = np.zeros((len(corpus), len(wd2idx)))
    for i, text in enumerate(corpus):
        all_features[i] = text_to_bow(text, wd2idx)
    return all_features


def write_prediction(out_file, predictions):

    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)



def split(data, labels, procentaj_valid=0.25):


    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    N = int((1 - procentaj_valid) * len(labels))
    train = data[indici[:N]]
    valid = data[indici[N:]]
    labels_train = labels[indici[:N]]
    labels_valid = labels[indici[N:]]
    return train, valid, labels_train, labels_valid


def cross_validate(k, data, labels):


    chunk_size = len(labels) // k  # il face int
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i + chunk_size:]])

        train = data[train_indici]
        valid = data[valid_indici]

        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

corpus = train_df['text']
labels = train_df['label'].values

toate_cuvintele = get_corpus_vocabulary(corpus)

wd2idx, idx2wd = get_representation(toate_cuvintele, 500)

data = corpus_to_bow(corpus, wd2idx)

test_data = corpus_to_bow(test_df['text'], wd2idx)



from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

clf = SVC(kernel = "linear", C = 0.5)
scoruri=[]
MatriceSuma=np.zeros((2, 2))
for train,valid,labels_train,labels_valid in cross_validate(10,data,labels):


    clf.fit(train,labels_train)
    predictii=clf.predict(valid)
    scor=f1_score(labels_valid,predictii)
    scoruri.append(scor)

    C=confusion_matrix(labels_valid, predictii)
    MatriceSuma= MatriceSuma + C


print(MatriceSuma)

print(np.mean(scoruri))



predictii = clf.predict(test_data)
write_prediction('submis.csv', predictii)

print("timp : %.2f "%(time.time()-timp_inceput))