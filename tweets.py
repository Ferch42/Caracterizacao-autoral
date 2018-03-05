import nltk
import logging
import sys
import os
import codecs

stopwords = nltk.corpus.stopwords.words('portuguese')

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

def func(list):
	s=''
	for word in list:		
		s=s+word+" "
	return s+'\n'
print("Tweets99")
with codecs.open('tweets_99.csv',encoding='utf8') as r, codecs.open('tweets_99_without_stopwords.csv','w',encoding='utf-8') as w:
	for line in r:
		w.write(func([word for word in line.replace('\n',' ').split() if word not in stopwords]))
			
print("Tweets150")
with codecs.open('tweets_150.csv',encoding='utf8') as r, codecs.open('tweets_150_without_stopwords.csv','w',encoding='utf-8') as w:
	for line in r:
		w.write(func([word for word in line.replace('\n',' ').split() if word not in stopwords]))
print("Tweetsfull")
with codecs.open('tweets_full.csv',encoding='utf8') as r, codecs.open('tweets_full_without_stopwords.csv','w',encoding='utf-8') as w:
	for line in r:
		w.write(func([word for word in line.replace('\n',' ').split() if word not in stopwords]))
	