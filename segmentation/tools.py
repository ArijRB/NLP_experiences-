import codecs
import re
import numpy as np
import nltk
import string
import unicodedata
import random
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation

    
    
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
    return np.array([re.sub('(?<!\w)[\d]+(?!\w)', '',doc) for doc in docs])
        
def build_vectorizer(docs, stopwords=None, b_stemming=False, b_lowercase=False, \
b_rmaccent=False, b_punctuation=False, b_rmnumbers=False, max_features=None, min_df=1, max_df=1.0):
    '''
        docs : liste de documents
        stopwords : liste de stopwords (None si gardés)  
        b_stemming : True si on effectue un stemming, False sinon
        b_lowercase : True si les mots sont mis en minuscules, False sinon
        b_punctuation : True si la ponctuation est gardée, False sinon
        b_rmaccent : True si les accents sont supprimés, False sinon
        max_features : si différent de None, nombre maximum de features à garder,
            dans l'ordre croissant de leur fréquence dans le corpus
        construit et renvoie un CountVectorizer construit à partir des 
            paramètres ci-dessus
    '''
    tokenizer_ = None
    lower = False

    if (not min_df is None) and ((min_df < 0) or (min_df > 1)):
        min_df = 1 # absolute counts

    if (not max_df is None) and ((max_df < 0) or (max_df > 1)):
        max_df = 1.0 # proportion of documents
        
    if isinstance(min_df, float):
        print("Ignoring terms in the vocabulary that have a document frequency < {}".format(min_df))
    if max_df < 1.0:
        print("Ignoring terms in the vocabulary that have a document frequency > {}".format(max_df))

    if b_rmnumbers:
        print("Removing numbers")
        docs = remove_numbers(docs)
    
    if b_stemming:
        print("Stemming")
        stemmer = PorterStemmer()
        tokenizer = CountVectorizer().build_tokenizer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in tokenizer(doc))

        tokenizer_ = stemmed_words

        if not (stopwords is None):
            stemmed_stopwords = [stemmer.stem(t) for t in stopwords]
            stopwords = stemmed_stopwords
            
    if b_lowercase:
        print("Removing uppercase")
        lower = True
    if b_rmaccent:
        print("Removing accents")

        if not (stopwords is None):
            stopwords = [remove_accents(w) for w in stopwords]

    if not (max_features is None):
        print("Keeping the top {} occurring tokens".format(max_features))

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
    max_features=max_features, min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(docs)
   
    
    return vectorizer, X
    
def vectorize_docs(vectorizer, docs):
    '''
        features : vectorizer built beforehand
        docs : documents to vectorize
        given a vectorizer, vectorize documents
    '''
    X = vectorizer.transform(docs)
    return X

