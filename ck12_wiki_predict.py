import argparse
import utils
import numpy as np
from scipy import linalg
import pickle
import pandas as pd
from nltk.corpus import stopwords

import gensim, logging, os
path = '/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/wiki_data'

stop = stopwords.words('english')

class MySentences(object):
    def __init__(self,dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            
            for line in open(os.path.join(self.dirname, fname)):
                yield utils.tokenize(line)

sentences = MySentences(path)

N = 1000
#model = gensim.models.Word2Vec(sentences, workers=3, size=N)

#model.save('concepts_and_wiki_data_%s' % N)
model = gensim.models.Word2Vec.load('concepts_and_wiki_data_%s' % N)

#print model['physics']
#print 'saved model'

#urls  to get toppics
"""'https://www.ck12.org/earth-science/', 'http://www.ck12.org/life-science/', 
                  'http://www.ck12.org/physical-science/', 'http://www.ck12.org/biology/', 
                  'http://www.ck12.org/chemistry/', """
ck12_url_topic = ['http://www.ck12.org/physics/'] #
wiki_docs_dir = 'data/wiki_data'


def get_wiki_docs():
    # get keywords 
    """
    for url_topic in ck12_url_topic:
        ck12_keywords = set()
        seg_dir = wiki_docs_dir + '/' + url_topic.split('/')[3]
        print seg_dir
        keywords= utils.get_keyword_from_url_topic(url_topic)
        for kw in keywords:
            ck12_keywords.add(kw)
    
        #get and save wiki docs
        utils.get_save_wiki_docs(ck12_keywords, save_folder=seg_dir)"""
    ck12_keywords = set()
    with open("/home/evan/Desktop/wiki_kw.txt", 'r') as f:
    	for line in f:
    		ck12_keywords.add(line.rstrip())
    utils.get_save_wiki_docs(ck12_keywords, save_folder=wiki_docs_dir)



def predict(data, docs_per_q):  
    #index docs
    docs_tf, words_idf = utils.get_docstf_idf(wiki_docs_dir)
    #docs_tf = pickle.load(open('docs_tf_data.p', 'rb'))
    #words_idf = pickle.load(open('words_idf_data.p', 'rb'))
    pickle.dump(docs_tf, open("docs_tf_data.p",  'wb'))
    pickle.dump(words_idf, open("words_idf_data.p", 'wb'))
    
    res = []
    print 'predict'
    for index, row in data.iterrows():
        #get answers words
        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))
        
        A_vec = np.zeros(N)
        B_vec = np.zeros(N)
        C_vec = np.zeros(N)
        D_vec = np.zeros(N)

        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0
        
        print index
        q = row['question']

        q_vec = np.zeros(N)
        for w in utils.tokenize(q):
            if w in model.vocab and w not in stop:
                q_vec += model[w]
        q_vec = q_vec / linalg.norm(q_vec)
        
        for d in zip(*utils.get_docs_importance_for_question(q, docs_tf, words_idf, docs_per_q))[0]:
            for w in w_A:
                if w in docs_tf[d]:      
                    sc_A += 1. * docs_tf[d][w] * words_idf[w] 
                    if w in model.vocab:
                        A_vec += model[w]# docs_tf (arr of tf for each doc for each word) [d] for the specific word 
            for w in w_B:
                if w in docs_tf[d]:
                    sc_B += 1. * docs_tf[d][w] * words_idf[w]
                    if w in model.vocab:
                        B_vec += model[w]
            for w in w_C:
                if w in docs_tf[d]:
                    sc_C += 1. * docs_tf[d][w] * words_idf[w]
                    if w in model.vocab:
                        C_vec += model[w]
            for w in w_D:
                if w in docs_tf[d]:
                    sc_D += 1. * docs_tf[d][w] * words_idf[w]
                    if w in model.vocab:
                        D_vec = model[w]

        A_vec = A_vec / linalg.norm(A_vec) 
        B_vec = B_vec / linalg.norm(B_vec)
        C_vec = C_vec / linalg.norm(C_vec)
        D_vec = D_vec / linalg.norm(D_vec)
        semantic_scores = np.concatenate((A_vec, B_vec, C_vec, D_vec)).reshape(4, N).dot(q_vec)
        semantic_scores[np.isnan(semantic_scores)] = 0
        #print semantic_scores
        combined_scores = [sc_A, sc_B, sc_C, sc_D] + semantic_scores
        #print combined_scores 
        res.append(['A','B','C','D'][np.argmax(combined_scores)])
        
    return res

if __name__ == '__main__':
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='validation_set.tsv', help='file name with data')
    parser.add_argument('--docs_per_q', type=int, default=20, help='number of docs to consider when ranking quesitons')
    parser.add_argument('--get_data', type=int, default= 0, help='flag to get wiki data for IR')
    args = parser.parse_args()
    
    
    if args.get_data:
        get_wiki_docs()
    
    #read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )
    #predict

    res = predict(data, args.docs_per_q)
    #save result
    pd.DataFrame({'id': list(data['id']), 'correctAnswer': res})[['id', 'correctAnswer']].to_csv("prediction_wiki_data_ss.csv", index = False)
    


    
        
        
         
    
    
    
