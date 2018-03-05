from smart_open import smart_open
import pandas as pd
from sklearn.datasets import load_mlcomp
import numpy as np
from numpy import random
import gensim
import sys
import os
import nltk
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report

path= 'C:\\Users\\Fernando\\Desktop\\ESTUDOS IC\\TESTE IRIS\\'
os.environ['MLCOMP_DATASETS_HOME']= "C:\\Users\\Fernando\\Desktop\\ESTUDOS IC\\TESTE IRIS"

metadata=''
confusion=[]
classnames=[]

### Choose application

entry=int(sys.argv[1])

if(entry==1):
	path =path +'A\\age'
	confusion=np.array([[0,0,0],[0,0,0],[0,0,0]])
	metadata='age'	

elif(entry==2):
	path =path+'G\\gender'
	confusion=np.array([[0,0],[0,0]])
	metadata='gender'

elif(entry==3):
	path =path+'R\\relig'
	confusion=np.array([[0,0,0],[0,0,0],[0,0,0]])
	metadata='relig'

elif(entry==4):
	path =path+'T\\ti'
	confusion=np.array([[0,0],[0,0]])
	metadata='ti'

else:
	print('Tente um número entre 1 - 4')
	print('1-age, 2-gender, 3-relig, 4-ti')
	exit(0)



### Load files
'''
Classes = os.listdir(path)

I={}
T={}
i=0
tag=0

for c in Classes:
	
	innerpath=os.path.join(path,c)
	files = os.listdir(innerpath)
	
	for f in files:
		I[i]= open(os.path.join(innerpath,f)).read().split()
		T[i]=tag
		i=i+1

	tag=tag+1
'''
###

### Load files
print("Loading database")
age_train= load_mlcomp(metadata,metadata)
print (age_train.DESCR)
print("%d documents" %len(age_train.filenames))
print("%d categories" % len(age_train.target_names))

I={}
T={}
i=0
for f in age_train.filenames:
	I[i]=open(f).read()
	i=i+1
i=0
for f in age_train.target:
	T[i]=f
	i=i+1


### KFold 
K_fold=KFold(n_splits=10, shuffle=True)

### Labels and Predictions
test_labels=np.array([],'int32')
test_pred=np.array([],'int32')

### Loading Model
modelpath='C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\ICMC\\skip_s1000.txt'
model = KeyedVectors.load_word2vec_format(modelpath, unicode_errors="ignore")
model.init_sims(replace=True)

### Defining Word averaging 
def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer1_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

### Defining Word Averaging List
def  word_averaging_list(wv, text_indices):
    return np.vstack([word_averaging(wv, I[f]) for f in text_indices])

### Separating test from train
for train_indices, test_indices in K_fold.split(I):
	
	### Creating train instances
	X_train = word_averaging_list(model, train_indices)
	
	target_list=[]
	for f in train_indices:
		target_list.append(T[f])
	y_train=np.array(target_list)

	### Creating test instances
	X_test = word_averaging_list(model, test_indices)

	target_list=[]
	for f in test_indices:
		target_list.append(T[f])
	y_test=np.array(target_list)
	test_labels=np.append(test_labels,y_test)
	
	### Creating Classifier
	knn_naive_dv = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine' )

	### Fitting the data
	knn_naive_dv.fit(X_train, y_train)

	### Predicting the data
	pred = knn_naive_dv.predict(X_test)
	test_pred=np.append(test_pred,pred)

	### Plotting Confusion Matrix
	confusion += confusion_matrix(y_test, pred)

### Printing results
print('---------------------------------------------------------------------------------')
print(classification_report(test_labels,test_pred,target_names=age_train.target_names))
print("Confusion matrix:")
print(confusion)
