### Whole tutorial W2Vec ###

### Input function

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
import gc
import scipy.sparse as sp
import time
from sklearn.naive_bayes import MultinomialNB

path= 'C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\iso-iso\\post-large-iso-iso\\profiling-fernando'
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

# Loading database

print("Loading database")
age_train= load_mlcomp(metadata,metadata)
print (age_train.DESCR)
print("%d documents" %len(age_train.filenames))
print("%d categories" % len(age_train.target_names))

# Indexing files

def myTokenizer(s):
	return s.split()

I={}
T={}
i=0
for f in age_train.filenames:
	I[i]=myTokenizer(open(f).read())
	i=i+1
i=0
for f in age_train.target:
	T[i]=f
	i=i+1

### Tags

def Tags(indices_list):
	tags=[]	
	for n in indices_list:
		tags.append(T[n])	
	return tags

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

def Model_filename(spmd,md):
	return 'C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\ICMC\\'+spmd+'_'+md+'.txt'

### Classifiers

def Create_Classifier(name):
	if(name=='LogReg'):
		return linear_model.LogisticRegression(n_jobs=1, C=1e5)
	elif(name=='KNN'):
		return KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
	else:
		raise NameError('Classifier Unavailable')

def Train_Classifier(classifier, features, tags):
	classifier.fit(features,tags)

# Experiment

def Output_string(cls,clf):
	return 'C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\Resultados\\New Embeddings (shortwords)\\'+clf+'\\'+cls+'_'+'new_model'+'.txt'

def Classify(cls,clf):

	#Start moment 
	Start_moment=time.time()
	title='Classifying '+cls+' using W2Vec '+ 'using new model(trained without shortwords)'
	print(title)

	#Loading model
	print('Loading model')
	new_model = gensim.models.Word2Vec.load('C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\Models\\skip_s600_shortwords\\skip_s600_stopwords_shortwords.model')
	model=new_model.wv
	model.init_sims(replace=True)

	#Creating the K-fold cross validator
	K_fold=KFold(n_splits=10, shuffle=True)

	# Labels
	test_labels=np.array([],'int32')
	test_pred=np.array([],'int32')

	# Confusion Matrix
	confusion=[]
	if(cls=='age' or cls=='relig'):
		confusion=np.array([[0,0,0],[0,0,0],[0,0,0]])
	elif(cls=='gender' or cls=='ti'):
		confusion=np.array([[0,0],[0,0]])
	else:
		raise NameError('Confusion Matrix Unavailable')

	# The test
	print('Running .... =)')
	for train_indices, test_indices in K_fold.split(I):

		X_train= word_averaging_list(model, train_indices)
		Y_train= np.array(Tags(train_indices))

		X_test= word_averaging_list(model, test_indices)
		Y_test= np.array(Tags(test_indices))
		test_labels=np.append(test_labels,Y_test)

		classifier= Create_Classifier(clf)
		Train_Classifier(classifier,X_train,Y_train)

		pred= classifier.predict(X_test)
		test_pred=np.append(test_pred,pred)
		confusion += confusion_matrix(Y_test, pred)

	report=classification_report(test_labels,test_pred,target_names=age_train.target_names)

	print(report)
	print("Confusion matrix:")
	print(confusion)
	Finish_moment= time.time()
	tm="It took "+str((Finish_moment-Start_moment))+" seconds"
	print(tm)

	f=open(Output_string(cls,clf),'w+')
	
	f.write(title+'\n \n')
	f.write(report+'\n \n')
	f.write("Confusion Matrix: \n")
	f.write(np.array_str(confusion)+'\n \n')
	f.write(tm)
	f.close()

	gc.collect()
###########################################################################################################
###########################################################################################################

### The main
Classifiers=['LogReg','KNN']
for c in Classifiers:
	Classify(metadata,c)