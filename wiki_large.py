import gensim, bz2
id2word = gensim..corpora.Dictionary.load_from_text(bz2.BZ2File('_wordids.txt.bz2'))
mm = gensim.corpora.MmCorpus('_tfidf.mm')

lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1 )
lda.save('wiki_200000_lda.lda')