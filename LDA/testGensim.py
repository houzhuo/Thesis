from numpy import *
from sklearn import datasets
import logging
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# First, fix the verbosity of the logger. In this example we're logging only warnings,
# but for a better debug, uprint all the INFOs.
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
logging.root.level = logging.WARNING


# show some text
news_dataset = datasets.fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = news_dataset.data

print "In the dataset there are", len(documents), "textual documents"
print "And this is the first one:\n", documents[0]


# word token for Doc[1]
def tokenize(text):
    return [token for token in gensim.utils.simple_preprocess(text) if token not in gensim.parsing.preprocessing.STOPWORDS]
print "After the tokenizer, the previous document becomes:\n", tokenize(documents[0])


# Next step: tokenize all the documents and build a count dictionary,
# that contains the count of the tokens over the complete text corpus
processed_docs = [tokenize(doc) for doc in documents]
word_count_dict = gensim.corpora.Dictionary(processed_docs)
print
print "In the corpus there are", len(word_count_dict), "unique tokens"


# filter rare tokens
# word must appear more than 10 times, no more than 20 Documents
word_count_dict.filter_extremes(no_below=20, no_above=0.1)
print "After filtering, in corpus there are only", len(word_count_dict), "unique tokens"

#bags of words model
bag_of_word_corpus = [word_count_dict.doc2bow(perdoc) for perdoc in processed_docs]
bow_doc1 = bag_of_word_corpus[0]  # the bow of first doc
print
print "Bag of words representation of the first document(tuples are composed by token_id and multiplicity):\n", bow_doc1
print
for i in range(5):
    print "In the document, topic_id {} (word \"{}\") appears {} time[s]".format(bow_doc1[i][0], word_count_dict[bow_doc1[i][0]], bow_doc1[i][1])
print "..."


# LDA mono-core
lda_model = gensim.models.LdaModel(bag_of_word_corpus, num_topics=10, id2word=word_count_dict, passes=5 )
# list of the words(and their relative weights) for each topic:
_ = lda_model.print_topic(-1)
# print the topics composition ,and their scores.
for index, score in sorted(lda_model[bag_of_word_corpus[0]], key=lambda tup: -1*tup[1]):
    print "Score: {} \t Topic: {}".format(score, lda_model.print_topic(index, 10))


# unseen Documents
unseen_document = "In my spare time I either play badmington or drive my car"
print "The unseen document is composed by the following text:", unseen_document
print

bow_vector = word_count_dict.doc2bow(tokenize(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda  tup: -1*tup[1]):
    print "Score: {}\t Topic: {}". format(score, lda_model.print_topic(index,5))