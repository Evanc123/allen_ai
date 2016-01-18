from os import listdir
from os.path import isfile, join
import os
import gensim
import csv
import re
from nltk.corpus import stopwords 

stop = stopwords.words('english')

def tokenize(review, remove_stopwords = True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # 1. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    # 3. Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words



TaggedDocument = gensim.models.doc2vec.TaggedDocument


data_dir = '~/Desktop/Kaggle/allen/glove/kaggle_allen/'
question_file = '/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/targets/validation_set.tsv'

docLabels = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + 'data/wiki_segmented_large')) for f in fn]

question_text = []
question_id = []

with open(question_file) as csvfile:
	reader = csv.reader(csvfile, delimiter='\t')
	for row in reader:
		question_id.append(row[0])
		question_text.append(row[1] + ' ' +  row[2] + ' ' + row[3] + ' ' + row[4] + ' ' + row[5])
		


question_text = question_text[1:]
question_id = question_id[1:]


text_file = open("doc_seg_filenames.txt", "w")
for item in docLabels:
	text_file.write("%s\n" % item)
text_file.close()

#print docLabels

data = []
for doc in docLabels:
    with open(doc, 'r') as f:
    	data.append(f.read())

docLabels.extend(question_id)
data.extend(question_text)

	

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=tokenize(data[idx]),tags=[self.labels_list[idx]])

it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it)

model.save("doc2vec_segmented_withquestionstokenized.model")



#TODO make list that associates wiki pages with index's 