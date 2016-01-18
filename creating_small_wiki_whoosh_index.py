import whoosh
import os, os.path
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from whoosh.qparser import QueryParser
import uuid




schema = Schema(article_title=ID(stored=True),
				content = TEXT(stored=True))

if not os.path.exists("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7"):
    os.mkdir("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7")

ix = index.create_in("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh7", schema)

i_dir = "/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/wiki_segmented"

writer = ix.writer()

for subdir, dirs, files in os.walk(i_dir):
	for file in files:
		with open(os.path.join(subdir, file), 'r') as content_file:
			writer.add_document( article_title= unicode(uuid.uuid1()), content=unicode(content_file.read(), "utf-8").replace('\n', ' '))
			#print content_file.read()
		print os.path.join(subdir, file)
writer.commit()


"""
def predict_bm25f(data, docs_per_q):  
    #index docs
    
    
    res = []
    
    for index, row in data.iterrows():
    	#print row['id']
        #get answers words
        w_A = set(utils.tokenize(row['answerA']))
        w_B = set(utils.tokenize(row['answerB']))
        w_C = set(utils.tokenize(row['answerC']))
        w_D = set(utils.tokenize(row['answerD']))
    
        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0
    
        question = row['question'] #first thing to debug if not working

        qp = QueryParser("content", schema=schema, group=qparser.OrGroup)
        qp.add_plugin(qparser.FuzzyTermPlugin())
        qp.remove_plugin_class(qparser.PhrasePlugin)
        qp.add_plugin(qparser.SequencePlugin())
        q = qp.parse(unicode(question, 'utf-8'))
       
        #cp = qparser.CompoundsPlugin( AndMaybe="&~")
        with ix.searcher() as s, ix.searcher(weighting=scoring.BM25F()) as scoring_searcher_bm25f:
        	results = s.search(q)
        	#print results
        	for document in results:
				doc_parser = QueryParser("content", schema=schema)
				doc_q = doc_parser.parse(u"article_title:%s" % document['article_title'])
				for w in w_A:
					sc_A += scoring.BM25F().scorer(scoring_searcher_bm25f, 'content', w).score(doc_q.matcher(scoring_searcher_bm25f))
					print w
				for w in w_B:
					sc_B += scoring.BM25F().scorer(scoring_searcher_bm25f, 'content', w).score(doc_q.matcher(scoring_searcher_bm25f))
				for w in w_C:
					sc_C += scoring.BM25F().scorer(scoring_searcher_bm25f, 'content', w).score(doc_q.matcher(scoring_searcher_bm25f))
				for w in w_D:
					sc_D += scoring.BM25F().scorer(scoring_searcher_bm25f, 'content', w).score(doc_q.matcher(scoring_searcher_bm25f))
        	

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])
        
    return res



    #parsing input arguments
 

   #read data
data = pd.read_csv('data/' + 'validation_set.tsv', sep = '\t' )



res = predict_bm25f(data, 10)

#predict
#res = predict(data, args.docs_per_q)
#save result
pd.DataFrame({'id': list(data['id']), 'correctAnswer': res})[['id', 'correctAnswer']].to_csv("predictions_bm25f_whoosh.csv", index = False)


"""