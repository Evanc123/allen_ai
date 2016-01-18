import whoosh
import os, os.path
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
from whoosh.qparser import QueryParser
"""
schema = Schema(article_title=ID(stored=True),
				content = TEXT(stored=True))

ix = index.open_dir("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh")

qp = QueryParser ("content", schema=schema)

q = qp.parse(u"physics")
with ix.searcher() as s:
	results = s.search(q)
	print results[1]



"""
schema = Schema(article_title=ID(stored=True),
				content = TEXT(stored=True))

if not os.path.exists("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh2"):
    os.mkdir("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh2")

ix = index.create_in("/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/whoosh2", schema)

i_dir = "/home/evan/Desktop/Kaggle/allen/glove/kaggle_allen/data/wiki_segmented"

writer = ix.writer()

for subdir, dirs, files in os.walk(i_dir):
	for file in files:
		with open(os.path.join(subdir, file), 'r') as content_file:
			writer.add_document( content=unicode(content_file.read(), "utf-8").replace('\n', ' '))
			#print content_file.read()
		print os.path.join(subdir, file)
writer.commit()

"""
for each directory: #AA, BB etc
	for each docuemtn: #Each doc has multiple docs
		article_count = Count("<doc")
		for i in article_count:
			s.add_article( ) #regex magic until next doc>, will require some indexing magic 

	s.commit() #commit after each directory or n gigabytes   


Use BM25 to get top N documents, then use TF-IDF to rank across documents 
"""

"""import os                                                                                                             

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:                                                                                            
        files = os.walk(subdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                r.append(subdir + "/" + file)                                                                         
    return r              """


"""
with open('wiki00', 'r') as myfile:
    data=myfile.read().replace('\n', '')


from HTMLParser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print "Encountered a start tag:", tag
    def handle_endtag(self, tag):
        print "Encountered an end tag :", tag
    def handle_data(self, data):
        print "Encountered some data  :", data

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me!</h1></body></html>')
"""

wiki_data_dir = '/media/evan/Elements/downloads/extracted'

schema = Schema(article_title=ID(stored=True),
				content = TEXT(stored=True))

	