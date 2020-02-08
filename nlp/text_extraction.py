import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 200
#%matplotlib inline

corpus = ['The sky is blue and beautiful.',
          'Love this blue and beautiful sky!',
          'The quick brown fox jumps over the lazy dog.',
          "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
          'I love green eggs, ham, sausages and bacon!',
          'The brown fox is quick and the blue dog is lazy!',
          'The sky is very blue and the sky is very beautiful today',
          'The dog is lazy but the brown fox is quick!'    
]
labels = ['weather', 'weather', 'animals', 'food', 'food', 'animals', 'weather', 'animals']


print("Plain corpus text \n",corpus)

corpus = np.array(corpus)
corpus_df = pd.DataFrame({'Document': corpus,
                          'Category': labels})
corpus_df = corpus_df[['Document', 'Category']]
print("\n\nPlain corpus DF with Document Category as column names  \n")
print(corpus_df)

##=== Text pre-processing=standardizing and notrmalizing=
##=== It includes : removetag,specialchar,accentchar,expandcontraction,stemming,lemmatization,removestopwords,removewhitespace,lowercase,etc.

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(corpus)

print("\n\nNormalized Filtered corpus text as part of pre-processing text for feature engineering \n")
print(norm_corpus)

##===

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix = cv_matrix.toarray()
#print(cv_matrix)
# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
count_vector_df=pd.DataFrame(cv_matrix, columns=vocab)
print("\n\n Document feature vector'S' obtained using countVectorizer\n")
print(count_vector_df)

###===

# you can set the n-gram range to 1,2 to get unigrams as well as bigrams

bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_corpus)

bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
count_vector_only_bigram=pd.DataFrame(bv_matrix, columns=vocab)
print("\n\n Bigram Document feature vector'S' obtained using countVectorizer\n")
print(count_vector_only_bigram)

##===

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
tfidf_vector_df=pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
print("\n\n TFIDF Document feature vector'S' obtained using countVectorizer\n")
print(tfidf_vector_df)

##===

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
print("\n\n Similarity between Documents obtained as cosine\n")
print(similarity_df)