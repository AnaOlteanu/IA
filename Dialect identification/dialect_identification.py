import numpy as np
import nltk
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score,classification_report
from sklearn.metrics import confusion_matrix
import io

def read_train_data(): # citirea datelor de antrenare (text + chei si label-urile corespunzatoare)
    data = np.genfromtxt('data/train_samples.txt', encoding='utf-8', dtype=None, comments=None, delimiter="\t", names=('key', 'text'))
    keys = data['key']
    train_samples = data['text']
    data_2 = np.genfromtxt('data/train_labels.txt', encoding='utf-8', dtype=None, delimiter="\t", names=('key', 'label'))
    labels = data_2['label']
    return train_samples,keys,labels

def read_test_data(): # citirea datelor de test (text + chei)
    data = np.genfromtxt('data/test_samples.txt',encoding='utf-8', dtype=None, comments=None, delimiter="\t", names=('key', 'text'))
    keys = data['key']
    test_samples = data['text']
    return test_samples,keys

def read_validation(): # citirea datelor de validare (text + chei; label-urile asociate)
    data = np.genfromtxt('data/validation_samples.txt', encoding='utf-8', dtype=None, comments=None, delimiter="\t", names=('key', 'text'))
    keys = data['key']
    train_samples = data['text']
    data_2 = np.genfromtxt('data/validation_labels.txt', encoding='utf-8', dtype=None, delimiter="\t", names=('key', 'label'))
    labels = data_2['label']
    return train_samples, keys, labels


train_samples, train_keys, train_labels = read_train_data()
test_samples, test_keys = read_test_data()
validation_samples, validation_keys, validation_labels = read_validation()

def split_mol_ro(train_samples, train_labels): # imparte textul in propozitii romanesti si moldovenesti in functie de label
    mol = []
    ro = []
    for i in range(0,len(train_labels)):
        if train_labels[i] == 0:
            mol.append(train_samples[i])
        elif train_labels[i] == 1:
            ro.append(train_samples[i])

    return mol,ro

mol_sent, ro_sent = split_mol_ro(train_samples,train_labels)

def labeled_sent(mol_sent, ro_sent): # creeaza o lista de perechi (p, l)
                                        # p = propozitie ro/mol ; l = label-ul asociat
    p = []
    for s in mol_sent:
        pair = (s,'0')
        p.append(pair)
    for s in ro_sent:
        pair = (s,'1')
        p.append(pair)
    return p

train_samples = labeled_sent(mol_sent, ro_sent)

train_sent = [] # contine mai intai prop mold si dupa cele reomanesti
train_labels = [] # contine label-urile asociate
for el in train_samples:
    train_sent.append(el[0])
    train_labels.append(el[1])


def get_feat(sentence): # construieste un dictionar pt fiecare propozitie data
    dict = {}
    sentence = WhitespaceTokenizer().tokenize(sentence)  # imparte prop in cuvinte
    for w in sentence:
        dict[w] = 0
    for w in sentence:
        dict[w] += 1

    return dict

vectorizer = CountVectorizer(analyzer=get_feat) # obiectul de tip Countvectorizer  (word embedding)

train_data = vectorizer.fit_transform(train_sent) # antrenam datele pe baza dictionarului creat cu get_unigrams
# print(train_data.shape)

#print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names())

test_data = vectorizer.transform(test_samples) # codifica datele de test


# from numpy import set_printoptions
# set_printoptions(threshold=nltk.sys.maxsize) # ajuta la printarea intregului vector
# print(test_data.toarray()[0])


model = MultinomialNB() # clasificatorul folosit
model.fit(train_data, train_labels) # antrenam pe datele de train si labels
predict = model.predict(test_data) # facem predictii pt datele de test

valid_data = vectorizer.transform(validation_samples) # codifica datele de validare
predict_valid = model.predict(valid_data) # predictii pe valid

pv = [] # retine label-urile aflate in urma predictiei
for p in predict_valid:
    if p == '0':
        pv.append(0)
    else:
        pv.append(1)

print("F1 score")
print(f1_score(validation_labels, pv))
print("================================")
print("Classification report")
print(classification_report(validation_labels, pv))
print("================================")
print("Confusion matrix")
print(confusion_matrix(validation_labels,pv))

def make_prediction_file(predict): # scrie predictiile in fisier
    list = []
    # creeaza lista de perechi de tip (cheie, predictie)
    for i in range(len(predict)):
        pair = str(test_keys[i]) + "," + str(predict[i]) + "\n"
        list.append(pair)
    with io.open("predictions.txt", "a+", encoding="utf8") as file:
        file.write("id, label\n")
        for l in list:
            file.write(l)

make_prediction_file(predict)
