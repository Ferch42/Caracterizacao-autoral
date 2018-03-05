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
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

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

# Loading database

print("Loading database")
age_train= load_mlcomp(metadata,metadata)
print (age_train.DESCR)
print("%d documents" %len(age_train.filenames))
print("%d categories" % len(age_train.target_names))

# Indexing files
stopwords = nltk.corpus.stopwords.words('portuguese')

def myTokenizer(s):
	return s.split()

def myTaggenizer(words,tags):
    return TaggedDocument(words=words,tags=tags)

I={}
T={}
i=0
for f in age_train.target:
	T[i]=[f]
	i=i+1
i=0
for f in age_train.filenames:
	I[i]=myTaggenizer(myTokenizer(open(f).read()),T[i])
	i=i+1
DT={}
i=0
for f in age_train.target:
	DT[i]=f
	i=i+1
### Features

def Features(indices_list):
	return [I[f] for f in indices_list]

### Tags

def Tags(indices_list):
	tags=[]	
	for n in indices_list:
		tags.append(DT[n])	
	return tags

### Classifiers 

def Create_Classifier(name):
	if(name=='LogReg'):
		return linear_model.LogisticRegression(n_jobs=1, C=1e5)
	else:
		raise NameError('Classifier Unavailable')

def Train_Classifier(classifier, features, tags):
	classifier.fit(features,tags)


# Experiment

def Output_string(cls,clf):
	return 'C:\\Users\\Fernando\\Desktop\\DEEP LEARNING 2\\Resultados\\Doc2Vec\\'+cls+'.txt'

def Classify(cls,clf):

	#Start moment 
	Start_moment=time.time()
	title='Classifying '+cls+' using Doc2Vec '+clf
	print(title)

	#Creating the K-fold cross validator
	K_fold=KFold(n_splits=10, shuffle=True)

	# Labels
	test_labels=np.array([],'int32')
	test_pred=np.array([],'int32')
	knn_pred=np.array([],'int32')

	# Confusion Matrix
	confusion=[]
	confusion_knn=[]
	if(cls=='age' or cls=='relig'):
		confusion=np.array([[0,0,0],[0,0,0],[0,0,0]])
		confusion_knn=np.array([[0,0,0],[0,0,0],[0,0,0]])
	elif(cls=='gender' or cls=='ti'):
		confusion=np.array([[0,0],[0,0]])
		confusion_knn=np.array([[0,0],[0,0]])
	else:
		raise NameError('Confusion Matrix Unavailable')

	# The test
	print('Running .... =)')
	print('----------------------------------')
	zet=1
	for train_indices, test_indices in K_fold.split(I):
		
		## Training model
		t1=time.time()
		print('Training model',str(zet)+'°','iteration')
		doc2vec_model = Doc2Vec(Features(train_indices), workers=1, size=5, iter=20, dm=1)
		print('Done in '+str(time.time()-t1))

		## Train Features
		t2=time.time()
		print('Fitting X_train',str(zet)+'°','iteration')
		X_train=[doc2vec_model.infer_vector(doc.words, steps=20) for doc in Features(train_indices)]
		Y_train=np.array(Tags(train_indices))
		print('Done in '+str(time.time()-t2))

		## Test Features
		t3=time.time()
		print('Fitting X_test',str(zet)+'°','iteration')
		X_test=[doc2vec_model.infer_vector(doc.words, steps=20) for doc in Features(test_indices)]
		Y_test= np.array(Tags(test_indices))
		test_labels=np.append(test_labels,Y_test)
		print('Done in '+str(time.time()-t3))

		t4=time.time()
		print('Classifying',str(zet)+'°','iteration')
		classifier= Create_Classifier(clf)
		Train_Classifier(classifier,X_train,Y_train)
		print('Done in '+str(time.time()-t4))

		t5=time.time()
		print('Predicting',str(zet)+'°','iteration')
		pred= classifier.predict(X_test)
		test_pred=np.append(test_pred,pred)
		confusion += confusion_matrix(Y_test, pred)
		print('Done in '+str(time.time()-t5))
		print('----------------------------------')

		print('Doing KNN lul \o/')
		knn_test_predictions = [ doc2vec_model.docvecs.most_similar([pred_vec], topn=1)[0][0]  for pred_vec in X_test]
		knn_pred=np.append(knn_pred,knn_test_predictions)
		confusion_knn += confusion_matrix(Y_test,knn_test_predictions)
		print('----------------------------------')

		zet=zet+1

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
	######################################################################################################
	report=classification_report(test_labels,knn_pred,target_names=age_train.target_names)
	title='Classifying '+cls+' using Doc2Vec KNN'

	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print(report)
	print("Confusion matrix:")
	print(confusion_knn)
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

	f.write('\n---------------------------------------------------------- \n')
	f.write('---------------------------------------------------------- \n')

	f.write(title+'\n \n')
	f.write(report+'\n \n')
	f.write("Confusion Matrix: \n")
	f.write(np.array_str(confusion_knn)+'\n \n')


	f.close()

	gc.collect()

###########################################################################################################
###########################################################################################################

### The main
Classify(metadata,'LogReg')