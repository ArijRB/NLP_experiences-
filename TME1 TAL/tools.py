import numpy as np
from collections import Counter
from nltk.stem.porter import *
stemmer = PorterStemmer()

def load(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            #print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2: # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.split(" ")
            doc.append((mots[0],mots[1]))
    return listeDoc
def load_min(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            #print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2: # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.split(" ")
            doc.append((mots[0].lower(),mots[1]))
    return listeDoc
def load_stem(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            #print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2: # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.split(" ")
            doc.append((stemmer.stem(mots[0]),mots[1]))
    return listeDoc

def create_dict_from_docs(list_docs):
	docs_dict = dict()
	for doc in list_docs:
		for (w, tag) in doc:
			docs_dict[w] = tag
				
	return docs_dict
	
def eval_with_dict(docs_dict, list_docs):
	score = 0	
	n=0
	for doc in list_docs:
		for (w, tag) in doc:
			n+=1
			docs_dict.setdefault(w, "inconnu")
			if docs_dict[w]==tag:
				score+=1
	return score,n

def eval_with_dict_maj(docs_dict, list_docs):
    score = 0
    n=0
    majoritaire,_=Counter(list(docs_dict.values())).most_common(1)[0]
    for doc in list_docs:
        for (w, tag) in doc:
            n+=1
            docs_dict.setdefault(w,majoritaire)
            if docs_dict[w]==tag:
                score+=1
    return score,n

def eval_(allpreds, alldocs):
	score = 0
	n = 0
	for i in range(len(allpreds)):
		for j in range(len(allpreds[i])):
			n += 1
			if allpreds[i][j] == alldocs[i][j][1]:
				score += 1
	return score,n

def from_docs_to_words(list_docs):
	words = []
	for doc in list_docs:
		words.append(np.array(doc)[:,0].tolist())
	return words

def from_docs_to_sequences(list_docs, list_docsT):
	buf = [[m for m,c in d ] for d in list_docs]
	mots = []
	[mots.extend(b) for b in buf]
	mots = np.unique(np.array(mots))
	nMots = len(mots)+1 # mot inconnu

	mots2ind = dict(zip(mots,range(len(mots))))
	mots2ind["UUUUUUUU"] = len(mots)

	buf2 = [[c for m,c in d ] for d in list_docs]
	cles = []
	[cles.extend(b) for b in buf2]
	cles = np.unique(np.array(cles))
	cles2ind = dict(zip(cles,range(len(cles))))

	nCles = len(cles)

	allx  = [[mots2ind[m] for m,c in d] for d in list_docs]
	allxT = [[mots2ind.setdefault(m,len(mots)) for m,c in d] for d in list_docsT]

	allq  = [[cles2ind[c] for m,c in d] for d in list_docs]
	allqT = [[cles2ind.setdefault(c,len(cles)) for m,c in d] for d in list_docsT]
	
	return allx, allxT, allq, allqT, mots2ind, cles2ind
	
# allx: liste de sequences d'observations
# allq: liste de sequences d'etats
# N: nb etats
# K: nb observation

def learnHMM(allx, allq, N, K, initTo1=True):
    if initTo1:
        eps = 1e-5
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    else:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    for x,q in zip(allx,allq):
        Pi[int(q[0])] += 1
        for i in range(len(q)-1):
            A[int(q[i]),int(q[i+1])] += 1
            B[int(q[i]),int(x[i])] += 1
        B[int(q[-1]),int(x[-1])] += 1 # derniere transition
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    B = B/np.maximum(B.sum(1).reshape(N,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return Pi , A, B

def viterbi(x,Pi,A,B):
    T = len(x)
    N = len(Pi)
    logA = np.log(A)
    logB = np.log(B)
    logdelta = np.zeros((N,T))
    psi = np.zeros((N,T), dtype=int)
    S = np.zeros(T)
    logdelta[:,0] = np.log(Pi) + logB[:,x[0]]
    #forward
    for t in range(1,T):
        logdelta[:,t] = (logdelta[:,t-1].reshape(N,1) + logA).max(0) + logB[:,x[t]]
        psi[:,t] = (logdelta[:,t-1].reshape(N,1) + logA).argmax(0)
    # backward
    logp = logdelta[:,-1].max()
    S[T-1] = logdelta[:,-1].argmax()
    for i in range(2,T+1):
        S[T-i] = psi[int(S[T-i+1]),T-i+1]
    return S, logp , logdelta, psi
 