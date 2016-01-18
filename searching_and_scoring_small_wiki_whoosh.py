import whoosh
import os, os.path
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh import scoring 
import utils
import argparse
import uuid
import numpy as np
import pandas as pd
from whoosh import qparser
from whoosh.reading import TermNotFound
import string
schema = Schema(article_title=ID(stored=True),
				content = TEXT(stored=True))

ix = index.open_dir("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7")
print ix
with ix.searcher() as s:
	query = QueryParser("content", schema).parse("physics")
	results = s.search(query)
	print results

def predict_TF_IDF(data, docs_per_q):  
    #index docs
    exclude = set(string.punctuation)
    
    res = []
    
    for idx, row in data.iterrows():
    	print row['id']
        #get answers words
        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))
    
        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0
    
        q_punc = row['question'] #first thing to debug if not working
        question =  ''.join(ch for ch in q_punc if ch not in exclude)
        qp = QueryParser("content", schema=schema, group=qparser.OrGroup)
        qp.add_plugin(qparser.FuzzyTermPlugin())
        qp.remove_plugin_class(qparser.PhrasePlugin)
        qp.add_plugin(qparser.SequencePlugin())
        q = qp.parse(unicode(question, 'utf-8'))
        #q = qp.parse('physics')
        #cp = qparser.CompoundsPlugin( AndMaybe="&~")
        with ix.searcher() as s, ix.searcher(weighting=scoring.TF_IDF()) as scoring_searcher_tfidf:
            results = s.search(q, limit=docs_per_q)
            """
            u_id = unicode(uuid.uuid1())
            if not os.path.exists("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7/%s" % u_id):
                os.mkdir("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7/%s" % u_id)
            q_ix = index.create_in("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7/%s" % u_id, schema)
            q_writer = q_ix.writer()
            for document in results:
                q_writer.add_document(article_title=document['article_title'], content=document['content'])
            q_writer.commit()
            """
           	# with q_ix.searcher(weighting=scoring.TF_IDF()) as scoring_searcher_tfidf
            for document in results:
                doc_parser = QueryParser("content", schema=schema)
                doc_q = doc_parser.parse(u"article_title:%s" % document['article_title'])
                for w in w_A:
                    try:
                        sc_A += scoring.TF_IDF().scorer(scoring_searcher_tfidf, 'content', w).score(doc_q.matcher(scoring_searcher_tfidf))
                    except TermNotFound:
                        pass
                for w in w_B:
                    try:
                        sc_B += scoring.TF_IDF().scorer(scoring_searcher_tfidf, 'content', w).score(doc_q.matcher(scoring_searcher_tfidf))
                    except TermNotFound:
                        pass
                for w in w_C:
                    try:
                        sc_C += scoring.TF_IDF().scorer(scoring_searcher_tfidf, 'content', w).score(doc_q.matcher(scoring_searcher_tfidf))
                    except TermNotFound:
                        pass
                for w in w_D:
                    try:
                        sc_D += scoring.TF_IDF().scorer(scoring_searcher_tfidf, 'content', w).score(doc_q.matcher(scoring_searcher_tfidf))
                    except TermNotFound:
                        pass
        	

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])
        
    return res


if __name__ == '__main__':
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='validation_set.tsv', help='file name with data')
    parser.add_argument('--docs_per_q', type=int, default= 10, help='number of docs to consider when ranking quesitons')
    args = parser.parse_args()
    
       #read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )



    res = predict_TF_IDF(data, args.docs_per_q)

    #predict
    #res = predict(data, args.docs_per_q)
    #save result
    pd.DataFrame({'id': list(data['id']), 'correctAnswer': res})[['id', 'correctAnswer']].to_csv("predictions_TF_IDF_whoosh.csv", index = False)
    




"""doc_parser = QueryParser ("content", schema=schema)


doc_q = doc_parser.parse(u"article_title:%s" % article_id)

term = "physics"
with ix.searcher(weighting=scoring.TF_IDF()) as scoring_searcher_tfidf:
	scorer = scoring.TF_IDF().scorer(scoring_searcher_tfidf, 'content', term).score(q.matcher(scoring_searcher_tfidf))
	
	print scorer
	#results = searcher_tfidf.search(q)

   	#print results[1]['article_title']"""