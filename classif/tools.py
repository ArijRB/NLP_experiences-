import codecs
import re
import numpy as np
import nltk
import string
import unicodedata
import random
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
import os
import pandas as pd
from random import shuffle
def parse_ftrain_CM():
    '''
        parse le fichier d'entraînement de la tâche Chirac-Mitterand
    '''
    fname = "AFDpresidentutf8/corpus.tache1.learn.utf8"
    alltxts = []
    labs = []
    with open(fname, 'rb') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.decode('utf-8')
            lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",line)
            txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",line)
            if lab.count('M') >0:
                labs.append(-1)
            else:
                labs.append(1)
            alltxts.append(txt.replace("\n", ""))
            
    return alltxts, np.array(labs)
    
def parse_ftest(fname = "AFDpresidentutf8/corpus.tache1.test.utf8"):
    '''
        parse le fichier de test de la tâche Chirac-Mitterand
    '''
    alltxts = []
    labs = []
    with open(fname, 'rb') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.decode('utf-8')
            txt = re.sub(r"<[0-9]*:[0-9]*>(.*)","\\1",line)
            alltxts.append(txt.replace("\n", ""))
            
    return alltxts
    
def write_predictions_file(predictions,version):
    '''
        predictions : array de prédictions (-1, 1)
        version : numéro
        écrit les prédictions dans un fichier en remplaçant -1 par 'M', 1 par 'C'
    '''
    file = open("testfile_v" + version + ".txt","w") 
    res="\n".join([str(e) for e in predictions]).replace("-1","M").replace("1","C")
    file.write(res+"\n")
    file.close() 
def write_predictions_file2(predictions,version):
    '''
        predictions : array de prédictions (-1, 1)
        version : numéro
        écrit les prédictions dans un fichier en remplaçant 1 par 'M', -1 par 'C'
    '''
    file = open("testfile_v" + version + ".txt","w") 
    res="\n".join([str(e) for e in predictions]).replace("-1","C").replace("1","M")
    file.write(res+"\n")
    file.close()
def eval_on_final_test(vectorizer, model, data_train, y_train, data_test, \
version,l=0):
    '''
        vectorizer : CountVectorizer
        model : modèle d'apprentissage
        data_train : données d'apprentissage
        y_train : labels de data_train
        data_test : données de test
        vectorise les données, entraîne le modèle sur data_train, renvoie et 
            écrit dans un fichier les prédictions obtenues sur data_test
    '''
    X_train = vectorize_docs(vectorizer, data_train)
    X_test_f = vectorize_docs(vectorizer, data_test)
    model.fit(X_train,y_train)
    pred = model.predict(X_test_f)
    if l==0:
        write_predictions_file(pred, str(version))
    elif l==1:
        write_predictions_file2(pred, "movie"+str(version))
    return pred
    
def draw_subsample(data, labels, y_maj):
    '''
        data : toutes les données
        labels : tous les labels (-1,+1)
        y_maj : label de la classe majoritaire
        renvoie un sous-échantillon de la classe majoritaire (de même taille
            que l'ensemble des données de la classe minoritaire)
    '''
    y_min = -1
    if y_maj == -1:
        y_min = 1
    ind_all_maj = np.where(labels == y_maj)[0]
    ind_all_min = np.where(labels == y_min)[0]
    ind_sample_maj = random.sample(list(ind_all_maj), \
                    len(ind_all_min))
    sample_data = []
    sample_labels = []
    
    for i in ind_all_min:
        sample_data.append(data[i])
        sample_labels.append(labels[i])
    for i in ind_sample_maj:
        sample_data.append(data[i])
        sample_labels.append(labels[i])
    c = list(zip(sample_data, sample_labels))
    random.shuffle(c)
    
    sample_data, sample_labels = zip(*c)
    sample_data = list(sample_data)
    sample_labels = list(sample_labels)
    return sample_data, sample_labels
    
def french_stopwords():
    '''
        retourne les stopwords français de nltk
    '''
    fr_stopwords = nltk.corpus.stopwords.words('french')
    return fr_stopwords
def english_stopwords():
    '''
        retourne les stopwords anhlais de nltk
    '''
    an_stopwords = nltk.corpus.stopwords.words('english')
    return an_stopwords

def remove_punctuation(doc):
    '''
        enlève la ponctuation dans le document
    '''
    return ''.join([char for char in doc if char not in string.punctuation])

def remove_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)
def remove_numbers(docs):
    return np.array([re.sub('.*\d+.*', '',doc) for doc in docs])

def build_vectorizer(docs, stopwords=None, b_stemming=False, b_lowercase=True, \
b_accent=True, b_punctuation=False, max_f=None,l=0,n=(1,1), b_rmnumbers=True):
    '''
        docs : liste de documents
        stopwords : liste de stopwords (None si gardés)  
        b_stemming : True si on effectue un stemming, False sinon
        b_lowercase : True si les mots sont mis en majuscules, False sinon
        b_punctuation : True si la ponctuation est gardée, False sinon
        b_accent : True si les accents sont gardés, False sinon
        max_f : si différent de None, nombre maximum de features à garder,
            dans l'ordre croissant de leur fréquence dans le corpus
        construit et renvoie un CountVectorizer construit à partir des 
            paramètres ci-dessus
    '''
    tokenizer_ = None
    lower = True
    
    if b_stemming:
        print("Stemming")
        if l==0:
            stemmer = SnowballStemmer('french')
        elif l==1:
            stemmer = SnowballStemmer("english")
        tokenizer = CountVectorizer().build_tokenizer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in tokenizer(doc))

        tokenizer_ = stemmed_words

        if not (stopwords is None):
            stemmed_stopwords = [stemmer.stem(t) for t in stopwords]
            stopwords = stemmed_stopwords
    if b_rmnumbers:
        print("Removing numbers")
        docs = remove_numbers(docs)

    if not b_lowercase:
        print("Keeping uppercase")
        lower = False
    if not b_accent:
        print("Removing accents")

        if not (stopwords is None):
            stopwords = [remove_accents(w) for w in stopwords]

    if not (max_f is None):
        print("Keeping the top {} occurring tokens".format(max_f))

    def clean_doc(s):
        if b_punctuation:
            s = re.findall(r"[\w']+|[.,!?;]", s)
        else:
            s = s.split(" ")
        clean_s = []
        for token in s:
            if lower:
                token = token.lower()
            clean_s.append(token)
        clean_s = ' '.join(clean_s)
        return clean_s
        
    vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenizer_, \
    max_features=max_f,ngram_range=n)
    X = vectorizer.fit_transform(docs)
   
    
    return vectorizer
    
def vectorize_docs(vectorizer, docs):
    '''
        features : vectorizer built beforehand
        docs : documents to vectorize
        given a vectorizer, vectorize documents
    '''
    X = vectorizer.transform(docs)
    return X

def parse_movies():
    path = "AFDmovies/movies1000"
    dir_pos = path + "/pos/"
    dir_neg = path + "/neg/"
    files_pos = os.listdir(dir_pos)
    files_neg = os.listdir(dir_neg)
    review=[]
    label=[]
    for fname in files_pos:
        file = open(dir_pos + fname, "r", encoding="utf-8")
        content = file.readlines()[4:]
        content = ' '.join(content)
        review.append(content)
        label.append(+1)
    for fname in files_neg:
        file = open(dir_neg + fname, "r", encoding="utf-8")
        content = file.readlines()[4:]
        content = ' '.join(content)
        review.append(content)
        label.append(-1)
    return review,label