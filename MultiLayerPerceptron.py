### Whole tutorial ###

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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

path= 'C:\\Users\\Fernando\\Desktop\\ESTUDOS IC\\TESTE IRIS\\'
os.environ['MLCOMP_DATASETS_HOME']= "C:\\Users\\Fernando\\Desktop\\ESTUDOS IC\\TESTE IRIS"

metadata=''
confusion=[]
classnames=[]

### Choose application

entry=int(sys.argv[1])

if(entry==1):
	path =path +'A\\age'
	metadata='age'	

elif(entry==2):
	path =path+'G\\gender'
	metadata='gender'

elif(entry==3):
	path =path+'R\\relig'
	metadata='relig'

elif(entry==4):
	path =path+'T\\ti'
	metadata='ti'

else:
	print('Tente um número entre 1 - 4')
	print('1-age, 2-gender, 3-relig, 4-ti')
	exit(0)

print("Loading database")
age_train= load_mlcomp(metadata,metadata)
print (age_train.DESCR)
print("%d documents" %len(age_train.filenames))
print("%d categories" % len(age_train.target_names))

I={}
T=[]
i=0
for f in age_train.filenames:
	I[i]=open(f).read()
	i=i+1
i=0
for f in age_train.target:
	T.append(f)

### Creation of Vectorizers 

stopwords = nltk.corpus.stopwords.words('portuguese')

def my_tokenizer(s):
	return s.lower().split()

def Create_Vectorizer(name):
	if(name=='CountVec'):
		return CountVectorizer(analyzer="word", tokenizer=my_tokenizer,preprocessor=None,stop_words=stopwords , max_features=3000) 
	elif(name=='NGram'):
		return CountVectorizer(analyzer="char", ngram_range=([2,5]), tokenizer=None,preprocessor=None, max_features=3000)
	elif(name=='TFidf'):
		return TfidfVectorizer(min_df=2, tokenizer=my_tokenizer, preprocessor=None,stop_words=stopwords)
	else:
		raise NameError('Vectorizer not found')

def Vec_fit_transform_train(vectorizer, indices_list):
	return vectorizer.fit_transform(I[f] for f in indices_list)

def Vec_fit_transform_test(vectorizer, indices_list):
	return vectorizer.transform(I[f] for f in indices_list)

### Tags

def Tags(indices_list):
	tags=[]	
	for n in indices_list:
		tags.append(T[n])	
	return tags

### Classifiers 

def Create_Classifier(name):
	if(name=='LogReg'):
		return linear_model.LogisticRegression(n_jobs=1, C=1e5)
	elif(name=='KNN'):
		return KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine')
	elif(name=='NB'):
		return MultinomialNB(alpha=0.01)
	elif(name=='MLPerceptron'):
		return MLPClassifier()
	else:
		raise NameError('Classifier Unavailable')

def Train_Classifier(classifier, features, tags):
	classifier.fit(features,tags)

### Experiment

def Output_string(cls,vec,clf):
	return 'C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\Resultados\\Regular Classifiers (lower)\\'+clf+'\\'+vec+'\\'+cls+'_'+vec+'-'+clf+'.txt'

def Classify(cls,vec,clf):

	# Creating vectorizer 
	Start_moment=time.time()
	vectorizer=Create_Vectorizer(vec)

	#Creating the K-fold cross validator
	#K_fold=KFold(n_splits=10, shuffle=True)

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

	
	data=vectorizer.fit_transform(I[f] for f in range(len(I)))
	print(data.shape)
	X_train, X_test, Y_train, Y_test = train_test_split(data,T, test_size=0.33, random_state=42)
	if(clf=='MLPerceptron'):
		scaler= StandardScaler(with_mean=False)
		scaler.fit(X_train)
		X_train=scaler.transform(X_train)
		X_test=scaler.transform(X_test)

	classifier= Create_Classifier(clf)
	Train_Classifier(classifier,X_train,Y_train)

	pred= classifier.predict(X_test)
	test_pred=np.append(test_pred,pred)
	confusion += confusion_matrix(Y_test, pred)

	report=classification_report(Y_test,pred,target_names=age_train.target_names)
	title='Classifying '+cls+' using '+vec+'-'+clf+' with stopwords'

	print(title)
	print(report)
	print("Confusion matrix:")
	print(confusion)
	Finish_moment= time.time()
	tm="It took "+str((Finish_moment-Start_moment))+" seconds"
	print(tm)
	gc.collect()

'''
	f=open(Output_string(cls,vec,clf),'w+')
	
	f.write(title+'\n \n')
	f.write(report+'\n \n')
	f.write("Confusion Matrix: \n")
	f.write(np.array_str(confusion)+'\n \n')
	f.write(tm)
	f.close()
'''
	
###########################################################################################################
###########################################################################################################

### The main

Classify(metadata,'TFidf','MLPerceptron')

