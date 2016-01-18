import gensim 
import csv
import argparse
import utils
from nltk.corpus import stopwords 
import numpy as np
import re
from scipy import linalg
import pandas as pd

stop = set(stopwords.words('english'))
model = gensim.models.Doc2Vec.load('doc2vecwithquestionstokenized.model')
wiki_docs_dir = 'data/wiki_segmented'

def most_common(lst):
	return max(set(lst), key=lst.count)


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

def get_question_ids():
	

	question_file = '/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/targets/validation_set.tsv'
	doc_label_txt = open('output.txt', 'r')
	doc_labels = doc_label_txt.read().split('\n')
	question_ids = []
	#question_text = []

	with open(question_file) as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		for row in reader:
			question_ids.append(row[0])
			#question_text.append(row[1])
	#print min(question_ids)
	return question_ids #, question_text
			
def get_category_and_id_dict(question_ids):
	ids_and_categories = {}
	doc_tags = model.docvecs.doctags
	
	for q_id in question_ids[1:]: 
		category = ''
		"""
		q_vec = np.zeros(300)
		for w in tokenize(question_text[i]):
			if w.lower() in model and w.lower() not in stop:
				q_vec += model[w.lower()]
		q_vec = q_vec / linalg.norm(q_vec)
		"""
		
		#sims = model.docvecs.most_similar(q_vec) #best optioon
		sims = model.docvecs.most_similar(q_id, clip_end=1505)

		q_categories = []
		for sim_model in sims:
			
			doc_tag = model.docvecs.offset2doctag[sim_model[0]]
			#print doc_tag
			try:
				int(doc_tag)
				continue
			except ValueError:
				pass #print doc_tag

			q_categories.append(doc_tag.split('/')[10])
		most_common_category = most_common(q_categories)
		ids_and_categories[q_id] = most_common_category

	return ids_and_categories



def predict_segmented_tf_idf(data, docs_per_q, ids_and_categories):  
    #index docs
    
    
    res = []
    category_tf_idfs = {}
    for index, row in data.iterrows():


    	current_id = str(row['id'])
    	print current_id
    	current_category = ids_and_categories[current_id]

    	if category_tf_idfs.get(current_category) is None:
    		category_tf_idfs[current_category] = utils.get_docstf_idf(wiki_docs_dir + '/%s' % current_category)

    	docs_tf, words_idf = category_tf_idfs[current_category]

        #get answers words
        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))
    
        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0
    
        q = row['question']
        
        for d in zip(*utils.get_docs_importance_for_question(q, docs_tf, words_idf, docs_per_q))[0]:
            for w in w_A:
                if w in docs_tf[d]:
                    sc_A += 1. * docs_tf[d][w] * words_idf[w] # count of how many times in the document, times log(numberofdocs/word) for each word
            for w in w_B:
                if w in docs_tf[d]:
                    sc_B += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_C:
                if w in docs_tf[d]:
                    sc_C += 1. * docs_tf[d][w] * words_idf[w]
            for w in w_D:
                if w in docs_tf[d]:
                    sc_D += 1. * docs_tf[d][w] * words_idf[w]

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])
        
    return res

if __name__ == '__main__':
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='validation_set.tsv', help='file name with data')
    parser.add_argument('--docs_per_q', type=int, default= 10, help='number of docs to consider when ranking quesitons')
    parser.add_argument('--get_data', type=int, default= 0, help='flag to get wiki data for IR')
    args = parser.parse_args()
    
    if args.get_data:
        get_wiki_docs()
    
    
    #read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )

    q_ids = get_question_ids()
    
    id_and_category_dict = get_category_and_id_dict(q_ids)

    res = predict_segmented_tf_idf(data, args.docs_per_q, id_and_category_dict)

    #predict
    #res = predict(data, args.docs_per_q)
    #save result
    pd.DataFrame({'id': list(data['id']), 'correctAnswer': res})[['id', 'correctAnswer']].to_csv("predictions_segmented.csv", index = False)
    
